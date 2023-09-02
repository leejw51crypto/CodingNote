all:
	echo "OK"


makekey:
	openssl genpkey -algorithm RSA -out key.pem -aes256
	openssl req -new -x509 -key key.pem -out cert.pem -days 365
