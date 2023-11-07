all:
	wasm-pack build --target web
	$(MAKE) -C hello-world-wasm
	

copy:
	cp -Rf pkg ./hello-world-wasm/src

install:
	npx create-react-app hello-world-wasm
