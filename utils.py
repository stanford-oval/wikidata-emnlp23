from pymongo import MongoClient
import time
import requests
import readline

client = MongoClient("mongodb://localhost:27017/")
qid_name_mapping = client["wikidata"]["qid_naming_mapping"]

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
        # print("processing QID {}".format(qid))
        r = requests.get(url, params = {'format': 'json', 'query': query})
        r.raise_for_status()
        try:
            name = r.json()["results"]["bindings"][0]["label"]["value"]
            
            
            # print("Found {} with name {}".format(qid, name))
            qid_name_mapping.insert_one({
                    "qid": qid,
                    "name": name
                }
            )
        
            return name
        except Exception as e:
            return None

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

def input_with_prefill(prompt, prefill=''):
    """Provide input prompt with pre-filled text."""
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()  # Unset the hook after using it.

def input_user(prompt: str, prefill='') -> str:
    user_utterance = input_with_prefill(bcolors.OKCYAN + bcolors.BOLD + prompt, prefill)
    while (not user_utterance.strip()):
        user_utterance = input_with_prefill(bcolors.OKCYAN + bcolors.BOLD + prompt, prefill)
    print(bcolors.ENDC)
    return user_utterance.strip()

import os
import io
import json
from typing import Dict
import transformers
from jinja2 import Environment, FileSystemLoader, select_autoescape

jinja_environment = Environment(loader=FileSystemLoader('./'),
                  autoescape=select_autoescape(), trim_blocks=True, lstrip_blocks=True, line_comment_prefix='#')

DEFAULT_PAD_TOKEN = "[PAD]"


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_prompt_dict(prompt_format: str):
    if prompt_format == "simple":
        prompt_dict = {
            "prompt_input": "{instruction}\n\n{input}",
            "prompt_no_input": "{instruction}\n\n",
        }
    elif prompt_format == "alpaca":
        prompt_dict = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
    else:
        raise ValueError("Unknown value for --prompt_format")

    return prompt_dict


def apply_prompt_format(prompt_dict, example):
    if example.get("input", "") != "":
        return prompt_dict["prompt_input"].format_map(example)
    else:
        return prompt_dict["prompt_no_input"].format_map(example)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_tokens_if_needed(tokenizer, model):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )


def sort_by_output_and_input_length(eval_set):
    """
    Sorts in reverse order of the length of the "output" field
    Sorting in reverse helps detect if GPU memory is not enough, or batch size is too large, early on
    """
    sort_indices, sorted_eval = tuple(
        zip(
            *sorted(
                enumerate(eval_set), key=lambda x: (len(x[1]["output"]), len(x[1]["instruction"])+len(x[1]["input"])), reverse=True
            )
        )
    )
    return list(sort_indices), list(sorted_eval)


def invert_sort(sort_indices, array):
    """
    Has the opposite effect of `sort_by_output_and_input_length`
    """
    return list(tuple(zip(*sorted(zip(sort_indices, array))))[1])



def fill_template(template_file, prompt_parameter_values={}):
    template = jinja_environment.get_template(template_file)

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = '\n'.join([line.strip() for line in filled_prompt.split('\n')]) # remove whitespace at the beginning and end of each line
    return filled_prompt