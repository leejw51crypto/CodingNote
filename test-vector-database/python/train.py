import json

import numpy as np
import pandas as pd
import torch
import os 
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
COLLECTION_NAME="mybook"


with open(f"data.json", "r") as file:
    meditations_json = json.load(file)

items= meditations_json["data"]
print(len(items))
# iterate items, display title, url
for item in items:
    print(f"title={item['title']}, sentences={len(item['sentences'])}")


# Create collection
qdrant_host = os.environ.get('QDRANT_HOST')
print(qdrant_host)
client = QdrantClient(qdrant_host)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
    	size=384, 
    	distance=models.Distance.COSINE
    ),
)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


rows = []
for chapter in items:
    for sentence in chapter["sentences"]:
        rows.append(
            (
                chapter["title"],
                chapter["url"],
                sentence,
            )
        )

df = pd.DataFrame(
    data=rows, columns=["title", "url", "sentence"]
)

df = df[df["sentence"].str.split().str.len() > 15]
print(df)


model = SentenceTransformer("msmarco-MiniLM-L-6-v3", device)
vectors = []
batch_size = 512
batch = []

for doc in tqdm(df["sentence"].to_list()):
    batch.append(doc)
    
    if len(batch) >= batch_size:
        vectors.append(model.encode(batch))
        batch = []

if len(batch) > 0:
    vectors.append(model.encode(batch))
    batch = []
    
vectors = np.concatenate(vectors)

book_name = meditations_json["book_title"]
print(df)
print(f"df {df.shape} vectors {vectors.shape}")
print(f"payloads {df.shape[0]} vectors {vectors.shape[0]}")

ids=[i for i in range(df.shape[0])]
payloads=[
    {
        "text": row["sentence"],
        "title": row["title"] + f", {book_name}",
        "url": row["url"],
    }
    for _, row in df.iterrows()
]

print(len(ids),ids[:2],"\n...\n",ids[-2:])
print(len(payloads),payloads[:2],"\n...\n",payloads[-2:])
client.upsert(
    collection_name=COLLECTION_NAME,
    points=models.Batch(
        ids=ids,
        payloads=payloads,
        vectors=[v.tolist() for v in vectors],
    ),
)

print("uploaded")