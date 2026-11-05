import argparse
import json
import os
from typing import TextIO

import tqdm
from vllm import LLM, SamplingParams

from util.prompt_utils import PRIMER_STRING, fill_prompt, generate_prompts

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def save_batch(model, sampling_params, prompts, records, output_file: TextIO) -> int:
    results = model.generate(prompts, sampling_params)
    if len(results) != len(records):
        raise RuntimeError(
            f"Model returned {len(results)} results for {len(records)} prompts"
        )

    serialized_records = []
    for result, record in zip(results, records):
        if not result.outputs:
            raise RuntimeError(
                f"Model returned no completion for prompt {record.get('prompt_index')}"
            )

        completion = result.outputs[0].text
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
    parser.add_argument("--personas_file", type=str, help="Personas JSON file", required=True)
    parser.add_argument("--instructions_file", type=str, help="Instructions JSON file", required=True)
    parser.add_argument("--pct_questions_file", type=str, help="Location of PCT questions", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for generated responses", required=True)
    parser.add_argument("--model_id", type=str, help="The name of the model to use", default=MODEL_ID)
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts to generate at a time")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum completion length")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be greater than zero")
    if args.max_tokens < 1:
        parser.error("--max_tokens must be greater than zero")

    model = LLM(
        model=args.model_id,
        trust_remote_code=True,
        download_dir="data/models",
        tensor_parallel_size=1,
        dtype="half",
        max_model_len=8000,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    with open(args.pct_questions_file, encoding="utf-8") as file:
        pct_questions = [line.strip() for line in file if line.strip()]
    prompt_fields_list = list(
        generate_prompts(args.personas_file, args.instructions_file)
    )
    total_answers = len(prompt_fields_list) * len(pct_questions)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "responses.jsonl")
    batch_prompts = []
    batch_records = []
    prompt_index = 0
    with open(output_path, "w", encoding="utf-8") as output_file:
        with tqdm.tqdm(
            total=total_answers,
            desc="Generating answers",
            unit="answer",
        ) as progress:
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
                            "question": question,
                            "options": options,
                            **prompt_fields,
                        }
                    )
                    prompt_index += 1

                    if len(batch_prompts) == args.batch_size:
                        progress.update(
                            save_batch(
                                model,
                                sampling_params,
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
                        model, sampling_params, batch_prompts, batch_records, output_file
                    )
                )


if __name__ == "__main__":
    main()
