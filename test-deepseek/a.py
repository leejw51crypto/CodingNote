import requests
import json


def query_ollama(prompt: str) -> str:
    """
    Send a query to the Ollama API using the DeepSeek model
    """
    url = "http://localhost:11434/api/generate"

    data = {"model": "nezahatkorkmaz/deepseek-v3", "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    # Example prompt
    prompt = "What is the meaning of life?"

    print("Sending query to DeepSeek model...")
    print(f"Prompt: {prompt}\n")

    response = query_ollama(prompt)
    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
