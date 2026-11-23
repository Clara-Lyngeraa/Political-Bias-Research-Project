import argparse
import json
from pathlib import Path
import pandas as pd
from util.prompt_utils import extract_selection_from_answer

answer_map = { 
    'Strongly disagree': 0, 
    'Strongly Disagree': 0, 
    'strongly disagree': 0, 
    'Disagree': 1, 
    'disagree': 1,
    'Agree': 2,
    'agree': 2,
    'Strongly agree': 3,
    'Strongly Agree': 3,
    'strongly agree': 3,
    'None': -1
    }

def fix_and_consolidate_data(input_path: Path, output_path: Path, base_case: bool) -> None:

    invalid_responses = []

    for model_converted_results in input_path.glob("*"):
        print(f"model_converted_results: {model_converted_results}")
        for question_type in model_converted_results.glob("*"):
            print(f"question_type: {question_type}")

            model_name = str(model_converted_results).split("/")[-1]
            generation_mode = str(question_type).split("/")[-1]

            consolidated_data_dir = output_path/model_name/generation_mode
            print(f"consolidated_data_dir: {consolidated_data_dir}")

            if not consolidated_data_dir.exists():
                consolidated_data_dir.mkdir(parents=True, exist_ok=True)

            for file in question_type.glob("*"):
                model_df = []
                print(f"file : {file}")

                if "base" in file.name:
                    csv_filename = f"{model_name}_base"
                else:
                    csv_filename = model_name

                contents = open(file, "r", encoding="utf-8").readlines()
                for line in contents:
                    content_json = json.loads(line)
                    if generation_mode == "closed":
                        response = extract_selection_from_answer(content_json)
                    else:
                        response = content_json["selection"]


                    if response not in list(answer_map.keys()):
                        invalid_responses.append((file, response))
                        continue

                    assert response in list(answer_map.keys()), f"Invalid response: {response} in file: {file}"
                    content_json["selection"] = response
                    content_json["model_name"] = model_name

                    model_df.append(content_json)

                model_df = pd.DataFrame(model_df)
                model_df.to_csv(f"{consolidated_data_dir}/{csv_filename}.csv", index=False)
    print(invalid_responses)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="")
    parser.add_argument("--output_path", type=str, required=True, help="")
    parser.add_argument("--base_case", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    fix_and_consolidate_data(input_path, output_path, args.base_case)
    fix_and_consolidate_data(input_path, output_path, args.base_case)

if __name__ == "__main__":
    main()
            