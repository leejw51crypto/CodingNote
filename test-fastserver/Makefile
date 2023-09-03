all:
	echo "OK"

server:
	cargo run --example server ./

client:
	cargo run --example client https://localhost:4433/Cargo.toml

makekey:
	openssl genpkey -algorithm RSA -out key.pem -aes256
	openssl req -new -x509 -key key.pem -out cert.pem -days 365
