import json

import numpy as np
import pandas as pd
import torch

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from tqdm.notebook import tqdm

with open(f"data.json", "r") as file:
    meditations_json = json.load(file)

items= meditations_json["data"]
print(len(items))
# iterate items, display title, url
for item in items:
    print(f"title={item['title']}, sentences={len(item['sentences'])}")
