import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handle_data.json_to_csv import consolidate_data


class ConsolidateDataTests(unittest.TestCase):
    def test_exports_only_closed_domain_records_and_flattens_answers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "input"
            model_dir = input_dir / "model-a"
            model_dir.mkdir(parents=True)
            input_file = model_dir / "responses.jsonl"
            output_file = Path(temporary_directory) / "output" / "responses.csv"
            records = [
                {
                    "prompt_index": 3,
                    "model": "model-a",
                    "question": "Question?",
                    "options": ["Agree", "Disagree"],
                    "age": "25",
                    "instruction": {"type": "closed_domain", "text": "Choose."},
                    "answer": (
                        '{"selection": "strongly disagree", '
                        '"response": "Reason, with a comma."}'
                    ),
                    "completion": 'strongly disagree", "response": "Reason, with a comma."}',
                },
                {
                    "instruction": {"type": "open_domain"},
                    "answer": '{"response": "Not exported"}',
                },
            ]
            input_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            exported_count = consolidate_data(input_dir, output_file)

            with output_file.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(exported_count, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_file"], "model-a/responses.jsonl")
            self.assertEqual(rows[0]["instruction_type"], "closed_domain")
            self.assertEqual(rows[0]["options"], '["Agree", "Disagree"]')
            self.assertEqual(rows[0]["selection_raw"], "strongly disagree")
            self.assertEqual(rows[0]["selection"], "Strongly disagree")
            self.assertEqual(rows[0]["selection_score"], "0")
            self.assertEqual(rows[0]["selection_valid"], "True")
            self.assertEqual(rows[0]["response"], "Reason, with a comma.")

    def test_recovers_selection_and_reports_invalid_json_lines(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "responses.jsonl"
            output_file = Path(temporary_directory) / "responses.csv"
            record = {
                "instruction": {"type": "closed_domain"},
                "answer": '{"selection": "Agree", "response": "unfinished',
                "completion": 'Agree", "response": "unfinished',
            }
            input_file.write_text(
                "{not json}\n" + json.dumps(record) + "\n", encoding="utf-8"
            )
            errors = StringIO()

            with redirect_stderr(errors):
                exported_count = consolidate_data(input_file, output_file)

            with output_file.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(exported_count, 1)
            self.assertIn("responses.jsonl:1", errors.getvalue())
            self.assertEqual(rows[0]["selection"], "Agree")
            self.assertEqual(rows[0]["selection_score"], "2")
            self.assertEqual(rows[0]["answer_parse_status"], "recovered")


if __name__ == "__main__":
    unittest.main()
