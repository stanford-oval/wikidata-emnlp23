import argparse
import requests
import json
from pymongo import MongoClient
from tqdm import tqdm
import datetime
import atexit
import subprocess
from utils import fill_template
import time
import multiprocessing
import re
from urllib.parse import urlencode
from location_silei import location_search

client = MongoClient("mongodb://localhost:27017/")
webquestion_dev = client["wikidata-eval"]["dev"]
webquestion_test = client["wikidata-eval"]["test"]
qald_test = client["wikidata-eval"]["qald7_test"]
qald_train = client["wikidata-eval"]["qald7_train"]
name_to_pid_mapping = client["wikidata-eval"]["name_to_pid_mapping"]
qid_name_mapping = client["wikidata"]["qid_naming_mapping"]
sparql_results = client["sparql_results"]["sparql_results"]

def get_name_from_qid(qid):
    candidate = qid_name_mapping.find_one({"qid" : qid})
    # print(candidate)
    if candidate:
        return candidate["name"]
    
    else:
        time.sleep(1)
        # include the wd:Q part
        url = 'https://query.wikidata.org/sparql'
        query = '''
    SELECT ?label
    WHERE {{
    {} rdfs:label ?label.
    FILTER(LANG(?label) = "en").
    }}
        '''.format(qid)
        print("processing QID {}".format(qid))
        r = requests.get(url, params = {'format': 'json', 'query': query})
        r.raise_for_status()
        try:
            name = r.json()["results"]["bindings"][0]["label"]["value"]
            
            
            print("Found {} with name {}".format(qid, name))
            qid_name_mapping.insert_one({
                    "qid": qid,
                    "name": name
                }
            )
        
            return name
        except Exception as e:
            return None

SERVER_PORT = 6000
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def execute_sparql(query):
    if sparql_results.find_one({"sparql": query}):
        return sparql_results.find_one({"sparql": query})["results"]
    
    url = 'https://query.wikidata.org/sparql'
    try:
        r = requests.get(url, params = {'format': 'json', 'query': query}, timeout=30, headers={"User-Agent":"Wikidata VA Analysis, Stanford OVAL"})
        r.raise_for_status()
        
        if "boolean" in r.json():
            res = r.json()['boolean']
        else:
            res = r.json()["results"]["bindings"]
    
    except requests.exceptions.HTTPError as err:
        if r.status_code == 500 or r.status_code == 400:
            print("Caught 500 or 400 Server Error:", err)
            res = []
        elif r.status_code == 429:
            time.sleep(2)
            res = execute_sparql(query)
        else:
            raise  # Reraise the exception if it's not a 500 error
    except requests.exceptions.ReadTimeout:
        res = []
    except requests.exceptions.JSONDecodeError or json.decoder.JSONDecodeError:
        res = []
    except requests.exceptions.ConnectionError:
        res = []
    except KeyError:
        res = []

    try:
        sparql_results.insert_one({
            "sparql": query,
            "results": res
        })
    except Exception:
        pass

    return res
            
def go_through_eval_file(target_db, overwrite_existing=False):
    for i in target_db.find():
        print(i["id"])
        if "results" in i and i["results"] != [] and not overwrite_existing:
            continue
        
        res = execute_sparql(i["clean_sparql"])
        
        target_db.update_one({"_id": i["_id"]}, {
            "$set": {"results": res}
        })
        print(res)
        time.sleep(1)

def eval_file_stats(target_db):
    count = 0
    no_results = []
    for i in target_db.find():
        if i["results"] != []:
            count += 1
        else:
            no_results.append(i["id"])
    
    return count, no_results

def do_ned_for_dev(target_db, mode):
    if mode == "refined":
        from refined.inference.processor import Refined
        refined = Refined.from_pretrained(model_name="/data0/wikidata-workdir/models/refined",
                                    entity_set="wikidata",
                                    download_files=True,
                                    use_precomputed_descriptions=True)

        def refined_ned(utterance):
            spans = refined.process_text(utterance)
            output = set()
            for span in spans:
                if span.predicted_entity.wikidata_entity_id:
                    qid = span.predicted_entity.wikidata_entity_id
                    wikidata_name = get_name_from_qid("wd:" + qid)
                    if wikidata_name is not None:
                        output.add((wikidata_name, qid))
            
            return output    
        
        dev_set = list(target_db.find())
        for i in tqdm(dev_set):
            utterance = i["utterance"]
            pid_mapping_list = list(refined_ned(utterance))
            target_db.update_one({
                "_id": i["_id"]
            }, {
                "$set": {
                    "refined_ned_results": pid_mapping_list
                }
            })
    elif mode == "oracle":
        dev_set = list(target_db.find())
        pattern = r"wd:Q\d+"
        for i in tqdm(dev_set):
            utterance = i["utterance"]
            qid_list = re.findall(pattern, i["clean_sparql"])
            qid_list_tuples = [(get_name_from_qid(i), i.split(":")[1]) for i in qid_list]
            target_db.update_one({
                "_id": i["_id"]
            }, {
                "$set": {
                    "oracle_ned_results": qid_list_tuples
                }
            })
    else:
        raise ValueError

    

def evaluate_dev(server_address, mode, model_path, target_db, oracle_or_refined):

    # batch = []
    # j = 0
    dev_set = list(target_db.find())
    for i in tqdm(dev_set):
        found = False
        # if "predictions" in i:
        #     for prediction in i["predictions"]:
        #         if prediction["model_path"] == model_path:
        #             found = True
        #             break
                
        if found:
            print("prediction already exists for {}".format(i["_id"]))
            continue
            
        # find the gold sparql in the other collection
        gold_sparql = i["clean_sparql"]
        
        utterance = i["utterance"]
        if oracle_or_refined == "refined":
            pid_mapping_list = i["refined_ned_results"]
        else:
            pid_mapping_list = i["oracle_ned_results"]
            
        _input = fill_template('prompts/property-name-gen.input', {
            "query": utterance,
            "qid_list_tuples": pid_mapping_list
        })
        _instruction = fill_template('prompts/property-name-gen.instruction')
        
        prompt = [
            "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
        ]
        
        
        output = requests.post(
            url="http://127.0.0.1:{}/completions".format(server_address),
            json={
                "engine": "llama",
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 500,
                "top_p": 1,
                "stop": ['\n', '</s>'],
            },
        )
        
        print(bcolors.WARNING + output.json()["choices"][0]["text"] + bcolors.ENDC)
        print(bcolors.OKBLUE + gold_sparql + bcolors.ENDC)
        
        existing_predictions = i["predictions"] if "predictions" in i else []
        target_db.update_one({
            "_id": i["_id"]},
            {
                "$set": {
                    "predictions" : [{
                        "mode": mode,
                        "model_path": model_path,
                        "sparql": output.json()["choices"][0]["text"],
                    }] + existing_predictions
                }
            })           
    
server_process = None

def start_server(model_path):
    global server_process
    print(' '.join(["python", "/data0/wikidata-workdir/evaluation/src/inference_server.py", "--model_name_or_path", model_path, "--port", str(SERVER_PORT)]))
    server_process = subprocess.Popen(["python", "/data0/wikidata-workdir/evaluation/src/inference_server.py", "--model_name_or_path", model_path, "--port", str(SERVER_PORT)], stdout=subprocess.PIPE)

    while True:
        output = server_process.stdout.readline().strip().decode()
        # print(output)  # Optional: Print the output for debugging purposes
        # print(type(output))  # Optional: Print the output for debugging purposes

        if "Debug mode: off" in output:
            print("breaking out")
            time.sleep(1)
            break

def stop_server():
    if server_process and server_process.poll() is None:
        server_process.terminate()
        server_process.wait()


def execute_predicted_sparql(sparql):
    # first, let's replace the properties
    
    # if ("wdt:instance_of/wdt:subclass_of" in sparql):
    #     print("HELPPPP\n\n\n\n\n")
    
    # print(sparql)
    sparql = sparql.replace("wdt:instance_of/wdt:subclass_of", "wdt:P31/wdt:P279")
    # print(sparql)
    
    
    url = 'https://query.wikidata.org/sparql'
    extracted_property_names =  [x[1] for x in re.findall(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', sparql)]
    #print(extracted_property_names)
    pid_replacements = {}
    for replaced_property_name in extracted_property_names:
        if not name_to_pid_mapping.find_one({"name" : replaced_property_name}):
            
            i = replaced_property_name.replace('_', ' ').lower()
            pid_query = """
                SELECT ?property ?propertyLabel WHERE {
                ?property rdf:type wikibase:Property .
                ?property rdfs:label "%s"@en .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""% i
            
            time.sleep(1)
            response = requests.get(url, params={'format': 'json', 'query': pid_query})
            response.raise_for_status()
            data = response.json()
            if 'results' in data and 'bindings' in data['results'] and len(data['results']['bindings']) > 0:
                # Extract the property ID from the response
                property_id = data['results']['bindings'][0]['property']['value']
                property_id = property_id.replace('http://www.wikidata.org/entity/', '')
                
                print("inserting {} for {}".format(replaced_property_name, property_id))
                name_to_pid_mapping.insert_one({
                    "name": replaced_property_name,
                    "pid": property_id
                })
            else:
                # try querying https://www.wikidata.org/w/api.php?action=wbsearchentities&search=songwriter&language=en&limit=20&format=json&type=property
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbsearchentities",
                    "search": i,
                    "language": "en",
                    "limit": 20,
                    "format": "json",
                    "type": "property"
                }
                encoded_url = url + "?" + urlencode(params)
                # print(encoded_url)
                time.sleep(1)
                response = requests.get(encoded_url)
                data = response.json()
                
                if "search" in data and len(data["search"]) > 0:
                    property_id = data["search"][0]["id"]
                    print("inserting {} for {} by querying aliases for property".format(replaced_property_name, property_id))
                    name_to_pid_mapping.insert_one({
                        "name": replaced_property_name,
                        "pid": property_id
                    })
                else:
                    
                    print("CANNOT FIND PROPERTY: {} for SPARQL {}".format(replaced_property_name, sparql))
                    return [], sparql

        pid = name_to_pid_mapping.find_one({"name" : replaced_property_name})["pid"]
        pid_replacements[replaced_property_name] = pid
    
    def sub_fcn(match):
        prefix = match.group(1)
        value = match.group(2)
        
        return prefix + pid_replacements[value]
    
    sparql = re.sub(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', lambda match: sub_fcn(match), sparql)
        
    # next, we need to replace the domain entities
    extracted_entity_names =  [x[1] for x in re.findall(r'(wd:)([a-zA-PR-Z_0-9-]+)', sparql)]
    #print(extracted_entity_names)
    qid_replacements = {}
    for extracted_entity_name in extracted_entity_names:
        if extracted_entity_name in ["anaheim_ca"]:
            qid_name_mapping.delete_many({
                "name": extracted_entity_name
            })
        
        found = False
        for i in qid_name_mapping.find():
            if i["name"] == extracted_entity_name and "qid" in i:
                found = True
                qid_replacements[extracted_entity_name] = i["qid"]
            elif i["name"].lower().replace(' ', '_').replace('/','_').replace('-', '_') == extracted_entity_name and "qid" in i:
                found = True
                qid_replacements[extracted_entity_name] = i["qid"]
                
        if not found:
            try_location = location_search(extracted_entity_name.replace("_", " "))
            if try_location is not None:
                try_location = "wd:" + try_location
                print("inserting {} for {}".format(try_location, extracted_entity_name))
                qid_name_mapping.insert_one({
                    "name": extracted_entity_name,
                    "qid": try_location
                })
                qid_replacements[extracted_entity_name] = try_location
            else:
            
            
            # trying querying https://www.wikidata.org/w/api.php?action=wbsearchentities&search=governor%20of%20oregon&language=en&limit=20&format=json
            # url = "https://www.wikidata.org/w/api.php"
            # params = {
            #     "action": "wbsearchentities",
            #     "search": extracted_entity_name.replace("_", " "),
            #     "language": "en",
            #     "limit": 20,
            #     "format": "json"
            # }
            # encoded_url = url + "?" + urlencode(params)
            # response = requests.get(encoded_url)
            # data = response.json()
            # time.sleep(1)
            
            # if "search" in data and len(data["search"]) > 0:
            #     found_entity_id = "wd:" + data["search"][0]["id"]
            #     qid_replacements[extracted_entity_name] = found_entity_id
            #     print("inserting {} for {}".format(found_entity_id, extracted_entity_name))
            #     qid_name_mapping.insert_one({
            #         "name": extracted_entity_name,
            #         "qid": found_entity_id
            #     })
            # else:

                print("CANNOT FIND ENTITY: {} for SPARQL {}".format(extracted_entity_name, sparql))
                return [], sparql
    
    def sub_entity_fcn(match):
        value = match.group(2)
        return qid_replacements[value]
    
    sparql = re.sub(r'(wd:)([a-zA-PR-Z_0-9-]+)', lambda match: sub_entity_fcn(match), sparql)
        
    # finally, we can execute
    prediction_results = execute_sparql(sparql)
    # time.sleep(1)
    return prediction_results, sparql
        
def compare_results(res1, res2):
    # each is a list of results
    if type(res1) is bool or type(res2) is bool:
        return res1 == res2

    
    res1 = [list(x.values()) for x in res1]
    res2 = [list(x.values()) for x in res2]
    if (res1 == res2):
        return True
    else:
        # print(res1, res2)
        return False

def safe_divide(x, y):
    if x == 0 and y == 0:
        return 0
    return x / y

def execute_predictions(model_path, target_db, overwrite_existing=False, write_to_file=False):
    def print_results(i, prediction, final_sparql):
        print(i["utterance"])
        print(bcolors.WARNING + final_sparql + bcolors.ENDC)
        print(bcolors.OKBLUE + i["clean_sparql"] + bcolors.ENDC)
        if write_to_file:
            with open("prediction_res.tsv", "a+") as fd:
                fd.write('{}\t{}\t{}\t{}\t{}\n'.format(i["id"], i["utterance"], i["clean_sparql"], prediction ,final_sparql))
    
    exact_match = 0
    total = 0
    total_F1_score = 0
    for i in target_db.find():
        if i["results"] == []:
            # print(i["utterance"])
            # print(i["clean_sparql"])
            continue
            
        total += 1

        # see if we have an existing result:        
        found_prediction = None
        if not overwrite_existing and "prediction_results" in i:
            for existing_prediction in i["prediction_results"]:
                if model_path == existing_prediction["model_path"]:
                    found_prediction = existing_prediction
                    print("use existing results for {}".format(i["id"]))
                    break
        
        
        if found_prediction is not None and (not overwrite_existing or total < 800):
            if found_prediction["final_sparql"] == i["clean_sparql"] or compare_results(found_prediction["results"], i["results"]):
                exact_match += 1
            else:
                model_prediction = None
                for prediction in i["predictions"]:
                    if prediction["model_path"] == model_path:
                        model_prediction = prediction["sparql"]
                        break
                print_results(i, found_prediction["final_sparql"], model_prediction)
            prediction_res = found_prediction["results"]
        else:
            found = False
            for prediction in i["predictions"]:
                if prediction["model_path"] == model_path:
                    print("predicted: " + prediction["sparql"])
                    prediction_results, final_sparql = execute_predicted_sparql(prediction["sparql"])
                    
                    if final_sparql == i["clean_sparql"] or compare_results(prediction_results, i["results"]):
                        exact_match += 1
                    else:
                        print_results(i, prediction["sparql"], final_sparql)
                    
                    found = True
                    break
            
            if not found:
                print("{} no prediction".format(prediction))
                raise ValueError
                
            prediction_results_db = {
                "model_path": model_path,
                "final_sparql": final_sparql,
                "results": prediction_results
            }
            old_prediction_results = []
            if "prediction_results" in i:
                for old_prediction_result in i["prediction_results"]:
                    if old_prediction_result["model_path"] != model_path:
                        old_prediction_results.append(old_prediction_result)
            
            try:
                target_db.update_one({
                    "_id": i["_id"],
                }, {
                    "$set": {
                        "prediction_results": [prediction_results_db] + old_prediction_results
                    }
                })
            except Exception:
                pass
            
            prediction_res = prediction_results
            
        gold_res = i["results"]
        if type(gold_res) == bool or type(prediction_res) == bool:
            total_F1_score += 1 if gold_res == prediction_res else 0
        else:
            true_positive = [x for x in prediction_res if x in gold_res]
            false_positive = [x for x in prediction_res if x not in gold_res]
            false_negative = [x for x in gold_res if x not in prediction_res]
            
            precision = safe_divide(len(true_positive), len(true_positive) + len(false_positive))
            recall    = safe_divide(len(true_positive), len(true_positive) + len(false_negative))
            if precision + recall == 0:
                this_f1 = 0
            else:
                this_f1 = 2 * precision * recall / (precision + recall)
            total_F1_score += this_f1

        print("accuracy: {}/{} = {}".format(exact_match, total, exact_match/total))
        print("F1 = {}".format(total_F1_score / total))


def conversant_check(server_address):
        
    _input = "Monica S. Lam"
    _instruction = "Have you heard of this professor?"
    
    prompt = [
        "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
    ]
    
    
    output = requests.post(
        url="http://127.0.0.1:{}/completions".format(server_address),
        json={
            "engine": "llama",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 500,
            "top_p": 1,
            "stop": ['\n', '</s>'],
        },
    )
    
    print(bcolors.WARNING + output.json()["choices"][0]["text"] + bcolors.ENDC)


if __name__ == "__main__":
    
    # to run evaluation, first choose a model to run
    
    # best model for paper submission (webq + alpaca)
    # model_path = "/data0/fewshot_refined_recovery_alpaca-1/checkpoint-378" 
    
    # best model for qald 7 test (webq + qald7 train + alpaca)
    model_path = "/data0/qald7_train_refined_recovery_20_alpaca-1/checkpoint-390"
    
    # if results are already inside mongodb, then you do not need to run any predictions / results fetching
    # for the above model, you can simply do:
    execute_predictions(model_path, qald_test, overwrite_existing=False, write_to_file=True)
    
    
    # ==========================================================================================================
    # if you are evaluating a new model, then you should do thd following:
    
    # 1st: start the model server to listen to evaluation requests
    # start_server(model_path)
    # atexit.register(stop_server)
    
    # 2nd: get predictions for your target dataset with required NED data (mode is either "refined" or "oracle")
    # this will run through the target dataset set and record predictions from model
    # evaluate_dev(SERVER_PORT, model_path, model_path, qald_test, "refined")
    
    # 3rd, finally, get and execution results from the model and compare with existing results, compute statistics
    # execute_predictions(model_path, qald_test, overwrite_existing=True)