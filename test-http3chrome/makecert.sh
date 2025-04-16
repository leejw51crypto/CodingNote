openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out request.csr
openssl x509 -req -days 365 -in request.csr -signkey key.pem -out cert.pem
openssl x509 -in cert.pem -text -noout
