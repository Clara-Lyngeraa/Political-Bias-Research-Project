# Political-Bias-Research-Project
Mathilde and Claras repo

```bash
source ~/.venv-vllm-metal/bin/activate                  
```

```bash
python3 src/generate_pct_answers.py \
  --personas_file data/prompting/personas_small.json \
  --instructions_file data/prompting/instructions_small.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id meta-llama/Llama-3.2-1B-Instruct
```

ollama run using uv
```bash
uv run python3 src/generate_pct_answers.py \
  --personas_file data/prompting/personas.json \
  --instructions_file data/prompting/instructions.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id llama3:8b
```

Generated responses are written to `../data/output/responses.jsonl`, one JSON
record per prompt. Generation defaults to batches of 8 prompts and a maximum
response length of 512 tokens. Adjust these with `--batch_size` and `--max_tokens`.
