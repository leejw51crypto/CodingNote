all: compile build
	yarn run serve

compile:
	wasm-pack build

build:
	npx webpack --config webpack.config.js

nodejs:
	wasm-pack build --target nodejs
	
web:
	cd ./dist && python3 -m http.server
