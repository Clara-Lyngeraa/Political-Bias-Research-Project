import argparse
from email import parser
import json
import os
from typing import TextIO

import tqdm
# from vllm import LLM, SamplingParams
import ollama
from concurrent.futures import ThreadPoolExecutor

from util.prompt_utils import PRIMER_STRING, fill_prompt, generate_prompts, get_instruction_template

MODEL_ID = "llama3.2:1b"

def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")

client = ollama.Client(host="http://localhost:11434")

def generate_one(model_id: str, prompt: str, max_tokens: int) -> str:
    response = client.generate(
        model=model_id, 
        prompt=prompt, 
        raw=True,
        options={
            "temperature": 1.0,
            "num_predict": max_tokens,
            "num_ctx": 8000,
        })
    return response["response"]


def save_batch(model, max_tokens, executor, prompts, records, output_file: TextIO) -> int:
    completions = list(
        executor.map(
            lambda prompt: generate_one(model, prompt, max_tokens),
            prompts
        )
    )

    serialized_records = []
    for completion, record in zip(completions, records):
        instruction_type = record["instruction"]["type"]
        answer = PRIMER_STRING[instruction_type] + completion
        serialized_records.append(
            json.dumps(
                {**record, "completion": completion, "answer": answer},
                ensure_ascii=False,
            )
        )

    output_file.write("\n".join(serialized_records) + "\n")
    return len(serialized_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas_file", type=str, required=True, help="Personas JSON file")
    parser.add_argument("--instructions_file", type=str, required=True, help="Instructions JSON file")
    parser.add_argument("--pct_questions_file", type=str, required=True, help="Location of PCT questions")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for generated responses")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="HF model id (must be in INSTRUCTION_TEMPLATES)")
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts to generate at a time")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum completion length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to shard the model over")
    parser.add_argument("--dtype", type=str, default="half", help="half | bfloat16 | float16 | auto")
    parser.add_argument("--quantization", type=str, default=None, help="Optional vLLM quantization, e.g. fp8, awq")
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--download_dir", type=str, default="data/models")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be greater than zero")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be greater than zero")

    # Get_instructions_template
    get_instruction_template(args.model_id)

    # model = LLM(
    #     model=args.model_id,
    #     trust_remote_code=True,
    #     download_dir=args.download_dir,
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     dtype=args.dtype,
    #     quantization=args.quantization,
    #     max_model_len=args.max_model_len,
    # )
    # sampling_params = SamplingParams(
    #     max_tokens=args.max_tokens,
    #     temperature=1.0,
    # )

    with open(args.pct_questions_file, encoding="utf-8") as file:
        pct_questions = [line.strip() for line in file if line.strip()]
    prompt_fields_list = list(
        generate_prompts(args.personas_file, args.instructions_file)
    )
    total_answers = len(prompt_fields_list) * len(pct_questions)

    model_dir = os.path.join(args.output_dir, model_slug(args.model_id))
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "responses.jsonl")

    batch_prompts = []
    batch_records = []
    prompt_index = 0
    with open(output_path, "w", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            with tqdm.tqdm(total=total_answers, desc=f"Generating ({args.model_id})", unit="answer") as progress:
                for prompt_fields in prompt_fields_list:
                    for question in pct_questions:
                        prompt, options = fill_prompt(
                            question=question,
                            model=args.model_id,
                            **prompt_fields,
                        )
                        batch_prompts.append(prompt)
                        batch_records.append(
                            {
                                "prompt_index": prompt_index,
                                "model": args.model_id,
                                "question": question,
                                "options": options,
                                **prompt_fields,
                            }
                        )
                        prompt_index += 1

                        if len(batch_prompts) == args.batch_size:
                            progress.update(
                                save_batch(
                                    args.model_id,
                                    args.max_tokens,
                                    executor,
                                    batch_prompts,
                                    batch_records,
                                    output_file,
                                )
                            )
                            batch_prompts.clear()
                            batch_records.clear()

                if batch_prompts:
                    progress.update(
                        save_batch(
                            args.model_id,
                            args.max_tokens,
                            executor,
                            batch_prompts,
                            batch_records,
                            output_file,
                        )
                    )


if __name__ == "__main__":
    main()
