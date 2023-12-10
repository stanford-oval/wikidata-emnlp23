import requests
import json
from utils import fill_template
import time
from urllib.parse import urlencode
from mention_heuristics import location_search
import re
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
# TODO: specify your mongo client for caching
# property name -> pid cache
name_to_pid_mapping = None
# TODO: specify your mongo client for caching
# entity name -> qid cache
qid_name_mapping = None
# TODO: specify your mongo client for caching
# sparql -> sparql results cache
sparql_results = None

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

def clean_sparql(sparql):

    def replace_before_select(input_string):
        # Define a regular expression pattern to match everything before "SELECT" and include "SELECT"
        pattern = r".*?(SELECT)"
        
        # Use re.sub to replace the matched portion with "SELECT"
        result = re.sub(pattern, r"\1", input_string)
        
        return result

    return replace_before_select(sparql)

def execute_predicted_sparql(sparql):
    # first, let's replace the properties
    
    sparql = sparql.replace("wdt:instance_of/wdt:subclass_of", "wdt:P31/wdt:P279")
    
    url = 'https://query.wikidata.org/sparql'
    extracted_property_names =  [x[1] for x in re.findall(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', sparql)]
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
    qid_replacements = {}
    for extracted_entity_name in extracted_entity_names:
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
        return False

def safe_divide(x, y):
    if x == 0 and y == 0:
        return 0
    return x / y

def execute_predictions(predicted_results_path, gold_results_path):
    def print_results(utterance, gold_sparql, final_sparql):
        print(utterance)
        print(bcolors.WARNING + final_sparql + bcolors.ENDC)
        print(bcolors.OKBLUE + gold_sparql + bcolors.ENDC)
    
    exact_match = 0
    total = 0
    total_F1_score = 0
    
    with open(gold_results_path, "r") as fd:
        gold_results = json.load(fd)
    with open(predicted_results_path, "r") as fd:
        predicted_results = json.load(fd)
    
    for i in gold_results:
        total += 1
        
        matching_records = list(filter(lambda record: record["dev_set_id"] == i["id"], predicted_results))
        if len(matching_records) != 1:
            raise Exception(f"No prediction found or multiple predictions found for {i['id']}. One id must have only exactly one prediction.")
        prediction = matching_records[0]
        
        if "executable_sparql" in prediction and "results" in prediction:
            final_sparql = prediction["executable_sparql"]
            prediction_results = prediction["results"]
        else:
            prediction_results, final_sparql = execute_predicted_sparql(prediction["predicted_sparql"])
        
        clean_gold_sparql = clean_sparql(i["sparql"])
        if final_sparql == clean_gold_sparql or compare_results(prediction_results, i["results"]):
            exact_match += 1
        else:
            print_results(i["utterance"], clean_gold_sparql, final_sparql)
            
        gold_res = i["results"]
        if type(gold_res) == bool or type(prediction_results) == bool:
            total_F1_score += 1 if gold_res == prediction_results else 0
        else:
            true_positive = [x for x in prediction_results if x in gold_res]
            false_positive = [x for x in prediction_results if x not in gold_res]
            false_negative = [x for x in gold_res if x not in prediction_results]
            
            precision = safe_divide(len(true_positive), len(true_positive) + len(false_positive))
            recall    = safe_divide(len(true_positive), len(true_positive) + len(false_negative))
            if precision + recall == 0:
                this_f1 = 0
            else:
                this_f1 = 2 * precision * recall / (precision + recall)
            total_F1_score += this_f1

        print("accuracy: {}/{} = {}".format(exact_match, total, exact_match/total))
        print("F1 = {}".format(total_F1_score / total))

if __name__ == "__main__":
    execute_predictions("predicted_results/best.json", "WikiWebQuestions/dev.json")