#!/bin/bash

# Create directory for certificates if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/server.key 2048

# Generate a certificate signing request (CSR)
openssl req -new -key certs/server.key -out certs/server.csr -subj "/CN=localhost"

# Generate a self-signed certificate valid for 365 days
openssl x509 -req -days 365 -in certs/server.csr -signkey certs/server.key -out certs/server.crt

echo "Self-signed certificates generated in ./certs/ directory"
