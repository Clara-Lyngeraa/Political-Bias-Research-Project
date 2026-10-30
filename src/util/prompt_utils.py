from typing import Any, AnyStr, Dict, Iterator
import json

SYSTEM_OPENING = ""

DEMOGRAPHIC_TEMPLATE = "Please follow all instructions from the perspective of a person who is {demographic}. Please write all responses in English."

def generate_prompts(
        personas_file: AnyStr,
        instructions_file: AnyStr
        ) -> Iterator[Dict[str, Any]]:

    with open(personas_file) as file:
        personas_json = json.load(file)
    with open(instructions_file) as file:
        instructions_json = json.load(file)

    demographic_fields = (
        ("age", "age"),
        ("gender", "gender"),
        ("political_orientation", "political orientation"),
    )
    instruction_groups = (
        ("closed_domain", "closed-domain"),
        ("open_domain", "open-domain"),
    )

    for prompt_field, persona_field in demographic_fields:
        for value in personas_json[persona_field]:
            for instruction_type, instruction_field in instruction_groups:
                for instruction_text in instructions_json[instruction_field]:
                    yield {
                        prompt_field: value,
                        "instruction": {
                            "type": instruction_type,
                            "text": instruction_text,
                        },
                    }

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
    political_orientation = political_orientation if political_orientation == '' else f"has {political_orientation} political views"
    
    demographics = [demo for demo in [age, gender, political_orientation] if demo != '']
    if len(demographics) > 0:
        system_prompt += DEMOGRAPHIC_TEMPLATE + ' '
    
        demo_string = ' '.join(demographics)
        print(demo_string)
        system_prompt = system_prompt.replace("{demographic}", demo_string)
        print(system_prompt)
    return demographics
    
