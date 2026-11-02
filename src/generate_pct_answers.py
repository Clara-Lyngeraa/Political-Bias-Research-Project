import argparse
import os
import tqdm
from vllm import LLM, SamplingParams
from util.prompt_utils import generate_prompts, fill_prompt

# MODEL_ID = "google/gemma-3-1b-it"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--personas_file", type=str, help="Personas JSON file", required=True)
    parser.add_argument("--instructions_file", type=str, help="Instructions JSON file", required=True)
    parser.add_argument("--pct_questions_file", type=str, help="Location of PCT questions", required=True)
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save the generated text", required=True)
    parser.add_argument("--model_id", type=str, help="The name of the model to use", default=MODEL_ID)
    # parser.add_argument("--cache_dir", type=str, help="Cache directory for HF models", required=True)
    # parser.add_argument("--exclusion_file", type=str, help="Location of file with list of prompts to exclude", default=None)
    # parser.add_argument("--split_file", type=str, help="If specified, uses the prompts in a given split file", default=None)
    # parser.add_argument("--model_id", type=str, help="The name of the model to use", default='meta-llama/Llama-2-13b-chat-hf',
                        # choices=['meta-llama/Llama-2-13b-chat-hf', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.2', 'google/gemma-7b-it', 'HuggingFaceH4/zephyr-7b-beta', 'allenai/OLMo-7B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'])
    # parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    # parser.add_argument("--random_samples", type=int, help="Specify a set number of random prompts to use", default=-1)
    # parser.add_argument("--base_case", action="store_true", help="Generate for the base case (i.e. no demographics)")
    
    args = parser.parse_args()
    personas_file = args.personas_file
    instructions_file = args.instructions_file
    pct_questions_file = args.pct_questions_file
    output_dir = args.output_dir
    model_id=args.model_id

    model = LLM(
        model=model_id, 
        trust_remote_code=True, 
        download_dir="data/models", 
        tensor_parallel_size=1, 
        dtype="half",
        max_model_len=8000,
        )
    sampling_params = SamplingParams(
        max_tokens=8000,
        temperature=0.0
    )

    if not os.path.exists(f"{output_dir}/success"):
        os.makedirs(f"{output_dir}/success")
    if not os.path.exists(f"{output_dir}/fail"):
        os.makedirs(f"{output_dir}/fail")

    with open(pct_questions_file) as file:
        pct_questions = [line.strip() for line in file]

    print("here")
    prompts = generate_prompts(personas_file, instructions_file)

    samples = []
    options_list = []
    prompt_options = []
    for i, prompt_fields in enumerate(tqdm.tqdm(prompts)):
        for j, question in enumerate(pct_questions):
            prompt, options = fill_prompt(question=question, model=model_id, **prompt_fields)

            samples.append(prompt)
            options_list.append(options)
            prompt_options.append(prompt_fields)

    llm_results = []
    result = model.generate(samples, sampling_params)[0]
    answer = result.outputs[0].text
