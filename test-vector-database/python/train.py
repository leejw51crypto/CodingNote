import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

COLLECTION_NAME="mybook"
DATA_PATH = "data.json"
# http://localhost:6333
QDRANT_HOST = os.environ.get('QDRANT_HOST')
BATCH_SIZE = 512
MODEL_NAME = "msmarco-MiniLM-L-6-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_data(path):
    print(f"Loading data from {path}")
    with open(path, "r") as file:
        meditations_json = json.load(file)
    return meditations_json["data"]

def create_qdrant_client(host):
    print(f"Creating Qdrant client with host {host}")
    client = QdrantClient(host)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    return client

def prepare_data(items):
    print("Preparing data")
    rows = []
    for chapter in items:
        for sentence in chapter["sentences"]:
            rows.append((chapter["title"], chapter["url"], sentence))

    df = pd.DataFrame(data=rows, columns=["title", "url", "sentence"])
    df = df[df["sentence"].str.split().str.len() > 15]
    return df

def encode_sentences(df, model_name, device, batch_size):
    print("Encoding sentences")
    model = SentenceTransformer(model_name, device)
    vectors = []
    batch = []

    for doc in tqdm(df["sentence"].to_list()):
        batch.append(doc)
        if len(batch) >= batch_size:
            encoded=model.encode(batch)
            print(f"batch length {len(batch)}")
            print(f"encoded length {len(encoded)}")
            vectors.append(encoded)
            batch = []

    if len(batch) > 0:
        encoded=model.encode(batch)
        print(f"batch length {len(batch)}")
        print(f"encoded length {len(encoded)}")
        vectors.append(encoded)
        batch = []
    
    # [[1,2,3],[4,5,6]] => [1,2,3,4,5,6]
    vectors = np.concatenate(vectors)
    return vectors

def create_payloads(df, vectors):
    print("Creating payloads")
    ids=[i for i in range(df.shape[0])]
    payloads=[
        {
            "text": row["sentence"],
            "title": row["title"],
            "url": row["url"],
        }
        for _, row in df.iterrows()
    ]
    return ids, payloads

def upload_data(client, ids, payloads, vectors):
    print("Uploading data")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=ids,
            payloads=payloads,
            vectors=[v.tolist() for v in vectors],
        ),
    )

def main():
    items = load_data(DATA_PATH)
    client = create_qdrant_client(QDRANT_HOST)
    df = prepare_data(items)
    print("beging encoding")
    vectors = encode_sentences(df, MODEL_NAME, DEVICE, BATCH_SIZE)
    ids, payloads = create_payloads(df, vectors)
    upload_data(client, ids, payloads, vectors)
    print("Uploaded.")

if __name__ == "__main__":
    main()
