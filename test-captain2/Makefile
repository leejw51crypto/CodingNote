all:server
	echo "OK"

server:
	cargo run --example server 127.0.0.1:8080
	
client:
	@RNDSTR=$$(openssl rand -base64 6 | sed 's/[^a-zA-Z]//g'); \
	cargo run --example client 127.0.0.1:8080 $$RNDSTR

play:
	cargo run --example play
