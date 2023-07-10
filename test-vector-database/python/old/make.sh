URL=$QDRANT_HOST
curl -X PUT "$URL/collections/test_collection" \
    -H 'Content-Type: application/json' \
    --data-raw '{
        "vectors": {
          "size": 4,
          "distance": "Dot"
        }
    }'
