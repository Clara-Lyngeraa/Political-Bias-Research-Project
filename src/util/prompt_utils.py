from typing import AnyStr, Dict
from itertools import combinations, product
import json

SYSTEM_OPENING = ""

DEMOGRAPHIC_TEMPLATE = "Please follow all instructions from the perspective of a person who is {demographic}. Please write all responses in English."

def generate_prompts(
        personas_file: AnyStr,
        instructions_file: AnyStr,
        n_categories: int = 2
        ):

    print("in here")
    with open(personas_file) as file:
        personas_json = json.load(file)
    with open(instructions_file) as file:
        instructions_json = json.load(file)

    initial_combinations = list(product(personas_json['age'], personas_json['gender'], personas_json['political_orientation'], instructions_json['closed_domain'])) + \
                            list(product(personas_json['age'], personas_json['gender'], personas_json['political_orientation'], instructions_json['open_domain']))
    for comb in initial_combinations:
        out_comb = comb + (personas_json['party'][comb[3]][comb[4]],)
        yield out_comb

def fill_prompt(
        instruction: Dict,
        question: AnyStr,
        age: AnyStr = '',
        gender: AnyStr = '',
        cls: AnyStr = '',
        nation: AnyStr = '',
        political_orientation: AnyStr = '',
        party: AnyStr = '',
        model: AnyStr="llama",
        **kwargs
        ):
    system_prompt = SYSTEM_OPENING
    age = age if age == '' else age + ' years old'
    political_orientation = political_orientation if political_orientation == '' else f"has {political_orientation} politcal views"
    
    demographincs = [demo for demo in [age, gender, political_orientation] if demo != '']
    if len(demographincs) > 0:
        system_prompt += DEMOGRAPHIC_TEMPLATE + ' '
    
        demo_string = ' '.join(demographincs)
        print(demo_string)
        system_prompt = system_prompt.replace("{demographic}", demo_string)
        print(system_prompt)
    return demographincs
    

