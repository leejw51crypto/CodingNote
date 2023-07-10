import json

import numpy as np
import pandas as pd
import torch
import os 
import pandas as pd
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
COLLECTION_NAME="mybook"



# Create collection
qdrant_host = os.environ.get('QDRANT_HOST')
OPENAI_API_KEY = os.environ.get('MYOPENAI')
print(qdrant_host)
print(OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY

qdrant_client = QdrantClient(qdrant_host)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
retrieval_model = SentenceTransformer("msmarco-MiniLM-L-6-v3", device)


def build_prompt(question: str, references: list) -> tuple[str, str]:
    prompt = f"""
    You're Marcus Aurelius, emperor of Rome. You're giving advice to a friend who has asked you the following question: '{question}'

    You've selected the most relevant passages from your writings to use as source for your answer. Cite them in your answer.

    References:
    """.strip()

    references_text = ""

    for i, reference in enumerate(references, start=1):
        text = reference.payload["text"].strip()
        references_text += f"\n[{i}]: {text}"

    prompt += (
        references_text
        + "\nHow to cite a reference: This is a citation [1]. This one too [3]. And this is sentence with many citations [2][3].\nAnswer:"
    )
    return prompt, references_text


def ask(question: str):
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=retrieval_model.encode(question),
        limit=3,
        append_payload=True,
    )

    prompt, references = build_prompt(question, similar_docs)
    # return prompt, references
    return (prompt, references)

def ask2(question: str):
    similar_docs = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=retrieval_model.encode(question),
        limit=3,
        append_payload=True,
    )

    prompt, references = build_prompt(question, similar_docs)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
        temperature=0.2,
    )

    return {
        "response": response["choices"][0]["text"],
        "references": references,
    }

    
# read text from user
text = input("Enter your question: ")
response=ask(text)
a=response
# a is tuple, print first element of a
print(f"{a[0]}")
print("copy & paste to chatgpt")
