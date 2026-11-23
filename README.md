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
  --base_case \
  --model_id llama3.2:1b
```

```bash
uv run python3 src/handle_data/open_to_closed.py \
  --model_id llama3.2:1b \
  --input_path data/output \
  --output_path data/output/converted
```

```bash
uv run python3 src/consolidate_data.py \
  --input_path data/output/converted \
  --output_path data/output/consolidated \
```

```bash
uv run python3 scripts/sample_for_review.py \
  --input "data/output/converted/llama3:8b/open/responses.jsonl" \
  -n 200 --seed 42 --parts 2 --part 1 -o data/review_llama3_open_part1.pdf

uv run python3 scripts/sample_for_review.py \
  --input "data/output/converted/llama3:8b/open/responses.jsonl" \
  -n 200 --seed 42 --parts 2 --part 2 -o data/review_llama3_open_part2.pdf
```
```bash
uv run python3 scripts/sample_for_review.py \
  --input "data/output/converted/olmo-3:7b-instruct/open/responses.jsonl" \
  -n 200 --seed 43 --parts 2 --part 1 -o data/review_olmo_open_part1.pdf

uv run python3 scripts/sample_for_review.py \
  --input "data/output/converted/olmo-3:7b-instruct/open/responses.jsonl" \
  -n 200 --seed 43 --parts 2 --part 2 -o data/review_olmo_open_part2.pdf
```


Generated responses are written to `../data/output/responses.jsonl`, one JSON
record per prompt. Generation defaults to batches of 8 prompts and a maximum
response length of 512 tokens. Adjust these with `--batch_size` and `--max_tokens`.

Export the closed-domain responses from the JSONL files under `data/output`:

```bash
uv run python src/consolidate_data.py
```

The CSV is written to `data/output/closed_domain_responses.csv`. Open-domain
records are excluded; response metadata, the selected option, its ordinal score,
and the original completion are included. Invalid JSONL lines are reported and
skipped. Pass `--input PATH` and `--output PATH` to use different locations.
