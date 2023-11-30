all: build

build:
	wasm-pack build --target web
	cp -Rf pkg ./hello/src

test:
	wasm-pack test --firefox --headless

install:
	cargo install wasm-pack wasm-bindgen-cli 

create:
	npx create-react-app hello --template typescript
