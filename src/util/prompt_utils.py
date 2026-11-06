from copy import copy
import random
from typing import Any, AnyStr, Dict, Iterator
import json

SYSTEM_OPENING = ""
SYSTEM_CLOSING = ""

DEMOGRAPHIC_TEMPLATE = "Please follow all instructions from the perspective of a person who {demographic}. Please write all responses in English."

FORMAT_STRING = {
    'closed_domain': "Format the output to JSON with the fields 'selection' with the option that you select and 'response' containing your response in unformatted plain text.",
    'open_domain': "Format all outputs to JSON with the field 'response' containing your response in unformatted plain text."
}

OPTION_TEXT = ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']

# Official Llama-3 chat format (blank line after every header).
LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# The template your existing Llama-3.2 results were generated with.
# NOTE: only ONE newline after the system/user headers (the official format has two).
# Kept as-is so old runs stay reproducible.
LEGACY_LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# ChatML-style format (<|im_start|> / <|im_end|>), used by OLMo 3.
CHATML_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

INSTRUCTION_TEMPLATES = {
    "meta-llama/Meta-Llama-3-8B-Instruct": LEGACY_LLAMA3_TEMPLATE,
    "meta-llama/Llama-3.2-3B-Instruct": LEGACY_LLAMA3_TEMPLATE,
    "meta-llama/Llama-3.2-1B-Instruct": LEGACY_LLAMA3_TEMPLATE,
    "meta-llama/Llama-3.3-70B-Instruct": LLAMA3_TEMPLATE,
    "allenai/Olmo-3-7B-Instruct": CHATML_TEMPLATE,
}

PRIMER_STRING = {
    'closed_domain': '{ "selection": "',
    'open_domain': '{ "response": "'
}

def get_instruction_template(model: str) -> str:
    try:
        return INSTRUCTION_TEMPLATES[model]
    except KeyError:
        raise ValueError(
            f"No chat template registered for '{model}'. "
            f"Add it to INSTRUCTION_TEMPLATES in util/prompt_utils.py. "
            f"Supported: {', '.join(INSTRUCTION_TEMPLATES)}"
        ) from None

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
        ("nation", "nation"),
        ("political_orientation", "political_orientation"),
    )
    instruction_groups = (
        ("closed_domain", "closed_domain"),
        ("open_domain", "open_domain"),
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
        model: AnyStr= '',
        **kwargs
        ):
    system_prompt = SYSTEM_OPENING
    age = age if age == '' else f'is {age} years old'
    gender = gender if gender == '' else f'is {gender}'
    nation = nation if nation == '' else f'is from {nation}'

    political_orientation = political_orientation if political_orientation == '' else f"has {political_orientation} political views"

    demographics = [demo for demo in [age, gender, nation, political_orientation] if demo != '']
    if len(demographics) > 0:
        system_prompt += DEMOGRAPHIC_TEMPLATE + ' '
    
        demo_string = ' '.join(demographics)
        system_prompt = system_prompt.replace("{demographic}", demo_string)


    system_prompt += SYSTEM_CLOSING + ' '
    system_prompt += FORMAT_STRING[instruction['type']]

    instruction_prompt = instruction['text']
    instruction_prompt = instruction_prompt.replace("{proposition}", question)

    if instruction["type"] == "closed_domain":
        options = copy(OPTION_TEXT)
        random.shuffle(options)
        option_text = ', '.join(options)
        instruction_prompt = instruction_prompt.replace("{options}", option_text)
    else:
        options = OPTION_TEXT

    prompt = get_instruction_template(model)
    prompt = prompt.replace("{system_prompt}", system_prompt)
    prompt = prompt.replace("{instruction}", instruction_prompt)

    prompt += PRIMER_STRING[instruction["type"]]

    # print(f"prompt: {prompt}")
    # print(f"options: {options}")

    return prompt, options 
