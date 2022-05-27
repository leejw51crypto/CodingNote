all: release cprogram crun

debug:
	cargo build
	
release:
	cargo build --release

cprogram:
	g++ -o myprogram cmain.cc -std=c++14 ./target/release/libmylib.a -framework CoreFoundation
	
crun:
	./myprogram
