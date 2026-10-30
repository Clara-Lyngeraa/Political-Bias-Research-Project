from typing import AnyStr
from itertools import combinations, product
import json



def generate_prompts_subsample(
        personas_file: AnyStr,
        instructions_file: AnyStr,
        n_categories: int = 2
        ):
    
    with open(personas_file) as file:
        personas_json = json.load(file)
    with open(instructions_file) as file:
        instructions_json = json.load(file)

    # Get combinations
    fields = ['age', 'gender', 'political_orientation']
    initial_combinations = []
    for combo in combinations(fields, n_categories):
        print(combo)
        options_closed = {key:personas_json[key] for key in combo}
        print(options_closed)
        options_closed['instruction'] = [{'text': inst, 'type': 'closed_domain'} for inst in instructions_json['closed_domain']]
        print(options_closed)

        options_open = {key:personas_json[key] for key in combo}
        options_open['instruction'] = [{'text': inst, 'type': 'open_domain'} for inst in instructions_json['open_domain']]

        initial_combinations.extend(
            [dict(zip(options_closed.keys(), values)) for values in product(*options_closed.values())] +
            [dict(zip(options_open.keys(), values)) for values in product(*options_open.values())]
        )

    print(initial_combinations)
    return initial_combinations
