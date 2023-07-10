from qdrant_client import QdrantClient
import os
from qdrant_client.models import Distance, VectorParams
import numpy as np
from qdrant_client.models import PointStruct

qdrant_host = os.environ.get('QDRANT_HOST')
#qdrant_host = QdrantClient(host="localhost", port=6333)
#qdrant_client = QdrantClient(
#    url="https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333",
#    api_key="<your-api-key>",
#)

print(qdrant_host)
client = QdrantClient(qdrant_host)
client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)


vectors = np.random.rand(100, 100)
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]
)


query_vector = np.random.rand(100)
hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5  # Return 5 closest points
)
print(hits)