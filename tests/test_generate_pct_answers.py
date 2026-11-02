import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_pct_answers import MODEL_ID, save_batch
from util.prompt_utils import fill_prompt


class GeneratePctAnswersTests(unittest.TestCase):
    def test_default_model_has_a_matching_prompt_template(self):
        prompt, _ = fill_prompt(
            instruction={"type": "closed_domain", "text": "Opinion on {proposition}: {options}"},
            question="Test proposition",
        )

        self.assertEqual(MODEL_ID, "meta-llama/Llama-3.2-1B-Instruct")
        self.assertTrue(prompt.startswith("<|begin_of_text|>"))
        self.assertIn("Test proposition", prompt)
        self.assertTrue(prompt.endswith('{ "selection": "'))

    def test_save_batch_writes_every_completion_as_jsonl(self):
        class FakeModel:
            def generate(self, prompts, sampling_params):
                self.prompts = prompts
                return [
                    SimpleNamespace(outputs=[SimpleNamespace(text='Agree", "response": "Reason 1."}')]),
                    SimpleNamespace(outputs=[SimpleNamespace(text='Disagree", "response": "Reason 2."}')]),
                ]

        records = [
            {"prompt_index": 1, "question": "Question 1", "instruction": {"type": "closed_domain"}},
            {"prompt_index": 2, "question": "Question 2", "instruction": {"type": "closed_domain"}},
        ]
        output = StringIO()

        count = save_batch(FakeModel(), object(), ["prompt 1", "prompt 2"], records, output)

        saved_records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(count, 2)
        self.assertEqual(len(saved_records), 2)
        self.assertEqual(saved_records[0]["question"], "Question 1")
        self.assertTrue(saved_records[0]["answer"].startswith('{ "selection": "Agree'))
        self.assertEqual(saved_records[1]["completion"], 'Disagree", "response": "Reason 2."}')

    def test_save_batch_rejects_mismatched_result_count(self):
        class FakeModel:
            def generate(self, prompts, sampling_params):
                return []

        with self.assertRaises(RuntimeError):
            save_batch(FakeModel(), object(), ["prompt"], [{"prompt_index": 1}], StringIO())


if __name__ == "__main__":
    unittest.main()
