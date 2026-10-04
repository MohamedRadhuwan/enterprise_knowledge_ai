import os
import ollama

model_name = os.getenv("LLM_MODEL", "llama3.2:3b")
host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

client = ollama.Client(host=host)

try:
    response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Explain what an LLM is in 3 concise bullet points.",
            }
        ],
    )
    print("=== Ollama Response ===")
    print(response["message"]["content"])
except Exception as e:
    print(f"Ollama connection error: {e}")