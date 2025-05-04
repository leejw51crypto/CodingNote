.PHONY: all rust swift python cpp clean

all: rust swift python cpp

rust:
	@echo "Running Rust FlexBuffer example..."
	@cargo run

swift:
	@echo "Running Swift FlexBuffer example..."
	@cd swift_flexbuffer && swift FlexBufferExample.swift

python:
	@echo "Running Python FlexBuffer example..."
	@cd python_flexbuffer && python3 flexbuffer_example.py

cpp:
	@echo "Building and running C++ FlexBuffer example..."
	@cd cpp_flexbuffer && mkdir -p build && cd build && cmake .. && make && cd .. && ./flexbuffer_example

clean:
	@echo "Cleaning up..."
	@rm -rf target
	@rm -rf cpp_flexbuffer/build
	@rm -f cpp_flexbuffer/flexbuffer_example
	@echo "Done!"

help:
	@echo "FlexBuffer Examples Makefile"
	@echo "Available targets:"
	@echo "  all     - Run all examples (default)"
	@echo "  rust    - Run Rust example"
	@echo "  swift   - Run Swift example"
	@echo "  python  - Run Python example"
	@echo "  cpp     - Build and run C++ example"
	@echo "  clean   - Clean up build artifacts"
	@echo "  help    - Show this help message" 