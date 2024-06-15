all: ccompile zigcompile
	./main

zigcompile:
	zig build-exe main.zig -lc -L. -lexample -I$(shell pwd)/include

ccompile:
	gcc -c example.c -o example.o
	ar rcs libexample.a example.o