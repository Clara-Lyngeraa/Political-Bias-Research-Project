import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handle_data import open_to_closed


class OpenToClosedTests(unittest.TestCase):
    def test_requests_schema_constrained_json_and_saves_classification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_root = Path(temporary_directory) / "input"
            model_input = input_root / "llama3.2:1b"
            model_input.mkdir(parents=True)
            input_file = model_input / "responses.jsonl"
            input_file.write_text(
                json.dumps(
                    {
                        "question": "The proposition.",
                        "answer": '{"response": "I disagree."}',
                        "instruction": {"type": "open_domain"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected_response = {
                "Explanation": "The opinion disagrees with the proposition.",
                "Decision": "Disagree",
            }
            chat_response = SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(expected_response))
            )

            with patch.object(open_to_closed.ollama, "chat", return_value=chat_response) as chat:
                open_to_closed.open_to_closed(
                    "llama3.2:1b",
                    str(input_root),
                    str(Path(temporary_directory) / "output"),
                )

            self.assertEqual(chat.call_args.kwargs["format"], open_to_closed.CLASSIFICATION_FORMAT)
            output_file = (
                Path(temporary_directory)
                / "output"
                / "llama3.2:1b"
                / "open"
                / "responses.jsonl"
            )
            result = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertEqual(result["selection"], "Disagree")
            self.assertEqual(
                result["explanation"],
                "The opinion disagrees with the proposition.",
            )


if __name__ == "__main__":
    unittest.main()
