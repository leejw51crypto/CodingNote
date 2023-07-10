from qdrant_client import QdrantClient

# read env QDRANT_HOST
import os
qdrant_host = os.environ.get('QDRANT_HOST')
print(qdrant_host)
client = QdrantClient(qdrant_host)

