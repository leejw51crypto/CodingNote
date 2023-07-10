URL=$QDRANT_HOST
curl -L -X POST "$URL/collections/test_collection/points/search" \
    -H 'Content-Type: application/json' \
    --data-raw '{
        "vector": [0.2,0.1,0.9,0.7],
        "top": 3
    }' | jq
