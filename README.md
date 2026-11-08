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
uv run python src/generate_pct_answers.py \
  --personas_file data/prompting/personas_small.json \
  --instructions_file data/prompting/instructions_small.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id llama3.2:1b
```

Generated responses are written to `../data/output/responses.jsonl`, one JSON
record per prompt. Generation defaults to batches of 8 prompts and a maximum
response length of 512 tokens. Adjust these with `--batch_size` and `--max_tokens`.

Running other models:

```bash
# OLMo 3 7B: fits on one GPU
python3 src/generate_pct_answers.py \
  --personas_file data/prompting/personas.json \
  --instructions_file data/prompting/instructions.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id allenai/Olmo-3-7B-Instruct \
  --dtype bfloat16

# Llama 3.3 70B: needs sharding (or quantization)
python3 src/generate_pct_answers.py \
   --personas_file data/prompting/personas.json \
  --instructions_file data/prompting/instructions.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id meta-llama/Llama-3.3-70B-Instruct \
  --tensor_parallel_size 4 --dtype bfloat16
```

## Visualizing responses

Open [visualizations/political_compass_responses.ipynb](visualizations/political_compass_responses.ipynb)
in VS Code or Jupyter and run the cells. Select a Python kernel with `pandas`,
`numpy`, `matplotlib`, and `seaborn` installed. The notebook plots complete
generated profiles using the Political Compass site's published scoring key
and includes response-distribution charts.
