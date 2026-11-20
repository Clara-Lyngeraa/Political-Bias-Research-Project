import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


SELECTION_SCORES = {
    "strongly disagree": ("Strongly disagree", 0),
    "disagree": ("Disagree", 1),
    "agree": ("Agree", 2),
    "strongly agree": ("Strongly agree", 3),
    "none": ("None", -1),
}
SELECTION_PATTERN = re.compile(r'"selection"\s*:\s*"([^"]*)"', re.IGNORECASE)
COMPLETION_SELECTION_PATTERN = re.compile(
    r'^\s*(.*?)"\s*,\s*"response"\s*:', re.DOTALL
)
CSV_COLUMNS = [
    "source_file",
    "prompt_index",
    "model",
    "question",
    "options",
    "age",
    "gender",
    "nation",
    "political_orientation",
    "instruction_type",
    "instruction_text",
    "selection_raw",
    "selection",
    "selection_score",
    "selection_valid",
    "response",
    "answer_parse_status",
    "completion",
]


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_answer(record: dict[str, Any]) -> tuple[str, str, int | str, str, str]:
    answer = record.get("answer")
    completion = record.get("completion")
    response = ""
    selection_raw = ""
    status = "unavailable"

    if isinstance(answer, str):
        try:
            parsed_answer = json.loads(answer)
        except json.JSONDecodeError:
            parsed_answer = None
        else:
            if isinstance(parsed_answer, dict):
                raw_selection = parsed_answer.get("selection")
                if isinstance(raw_selection, str):
                    selection_raw = raw_selection.strip()
                raw_response = parsed_answer.get("response")
                if isinstance(raw_response, str):
                    response = raw_response
                status = "parsed"

        if not selection_raw:
            match = SELECTION_PATTERN.search(answer)
            if match:
                selection_raw = match.group(1).strip()
                status = "recovered"

    if not selection_raw and isinstance(completion, str):
        match = COMPLETION_SELECTION_PATTERN.match(completion)
        if match:
            selection_raw = match.group(1).strip()
            status = "recovered"

    normalized = selection_raw.casefold()
    canonical_selection, score = SELECTION_SCORES.get(normalized, ("", ""))
    return selection_raw, canonical_selection, score, response, status


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Input file must have a .jsonl extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(input_path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_path}")
    return files


def _source_name(file_path: Path, input_path: Path) -> str:
    if input_path.is_dir():
        return file_path.relative_to(input_path).as_posix()
    return file_path.name


def consolidate_data(input_path: Path, output_path: Path) -> int:
    """Write all closed-domain JSONL records under input_path to one CSV."""
    files = _iter_input_files(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for file_path in files:
            with file_path.open(encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        print(
                            f"Skipping invalid JSON in {file_path}:{line_number}: "
                            f"{error.msg}",
                            file=sys.stderr,
                        )
                        continue
                    if not isinstance(record, dict):
                        print(
                            f"Skipping non-object JSON in {file_path}:{line_number}",
                            file=sys.stderr,
                        )
                        continue

                    instruction = record.get("instruction")
                    if (
                        not isinstance(instruction, dict)
                        or instruction.get("type") != "closed_domain"
                    ):
                        continue

                    (
                        selection_raw,
                        selection,
                        selection_score,
                        response,
                        answer_parse_status,
                    ) = _extract_answer(record)
                    writer.writerow(
                        {
                            "source_file": _source_name(file_path, input_path),
                            "prompt_index": record.get("prompt_index", ""),
                            "model": record.get("model", ""),
                            "question": record.get("question", ""),
                            "options": _json_text(record.get("options")),
                            "age": record.get("age", ""),
                            "gender": record.get("gender", ""),
                            "nation": record.get("nation", ""),
                            "political_orientation": record.get(
                                "political_orientation", ""
                            ),
                            "instruction_type": instruction.get("type", ""),
                            "instruction_text": instruction.get("text", ""),
                            "selection_raw": selection_raw,
                            "selection": selection,
                            "selection_score": selection_score,
                            "selection_valid": bool(selection),
                            "response": response,
                            "answer_parse_status": answer_parse_status,
                            "completion": record.get("completion", ""),
                        }
                    )
                    row_count += 1

    return row_count


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export closed-domain responses from JSONL files to CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "output",
        help="A JSONL file or directory to search recursively (default: data/output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "output" / "closed_domain_responses.csv",
        help="CSV file to write (default: data/output/closed_domain_responses.csv)",
    )
    args = parser.parse_args()

    try:
        row_count = consolidate_data(args.input, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote {row_count} closed-domain responses to {args.output}")


if __name__ == "__main__":
    main()
