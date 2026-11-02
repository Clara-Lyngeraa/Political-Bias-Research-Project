# Political-Bias-Research-Project
Mathilde and Claras repo

```bash
python3 generate_pct_answers.py \
  --personas_file ../data/prompting/personas.json \
  --instructions_file ../data/prompting/instructions.json \
  --pct_questions_file ../data/pct/political_compass_questions.txt \
  --output_dir ../data/output \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  > ../output.txt 2>&1
```

Generated responses are written to `../data/output/responses.jsonl`, one JSON
record per prompt. Generation defaults to batches of 8 prompts and a maximum
response length of 512 tokens. Adjust these with `--batch_size` and `--max_tokens`.
