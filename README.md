# Political-Bias-Research-Project
Mathilde and Claras repo

```bash
source ~/.venv-vllm-metal/bin/activate                  
```

```bash
python3 src/generate_pct_answers.py \
  --personas_file data/prompting/personas.json \
  --instructions_file data/prompting/instructions.json \
  --pct_questions_file data/pct/political_compass_questions.txt \
  --output_dir data/output \
  --model_id meta-llama/Llama-3.2-1B-Instruct
```

Generated responses are written to `../data/output/responses.jsonl`, one JSON
record per prompt. Generation defaults to batches of 8 prompts and a maximum
response length of 512 tokens. Adjust these with `--batch_size` and `--max_tokens`.

## Visualizing responses

Open [visualizations/political_compass_responses.ipynb](visualizations/political_compass_responses.ipynb)
in VS Code or Jupyter and run the cells. Select a Python kernel with `pandas`,
`numpy`, `matplotlib`, and `seaborn` installed. The notebook plots complete
generated profiles using the Political Compass site's published scoring key
and includes response-distribution charts.
