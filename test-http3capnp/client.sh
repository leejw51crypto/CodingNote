#!/bin/bash
# Run the HTTP/3 client with GET requests
echo -e "\n== Testing GET / =="
python client.py --insecure https://127.0.0.1:4433/

echo -e "\n== Testing GET /hello/world =="
python client.py --insecure https://127.0.0.1:4433/hello/world

echo -e "\n== Testing GET /gettime =="
python client.py --insecure https://127.0.0.1:4433/gettime

# Example of POST request
echo -e "\n== Testing POST /echo =="
python client.py --insecure  --method POST --data "Hello from HTTP/3 client!" https://127.0.0.1:4433/echo

echo -e "\n== Testing GET /capnp/hello (Cap'n Proto)=="
python client.py --insecure --capnp https://127.0.0.1:4433/capnp/hello

echo -e "\n== Testing POST /capnp/hello (Cap'n Proto)=="
python client.py --insecure --capnp --method POST --data "Lee" https://127.0.0.1:4433/capnp/hello
