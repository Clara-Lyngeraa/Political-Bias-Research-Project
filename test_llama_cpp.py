from llama_cpp import Llama

MODEL_PATH = ""

llm = Llama(
    model_path = MODEL_PATH,
    n_ctx = 2048,
    verbose = False
)

response = llm.create_chat_completion(
    messages = [
        {"role": "user", "content": "What is this?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])