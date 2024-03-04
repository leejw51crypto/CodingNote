// hello_llama.cpp
#include <iostream>
#include "llama.h"

int main() {
    std::cout << "Initializing llama backend..." << std::endl;

    // Initialize the llama backend
    llama_backend_init();

    // For this simple example, we won't do anything other than initialize and
    // free the backend.
    std::cout << "Llama backend initialized successfully." << std::endl;

    // Free the llama backend
    llama_backend_free();

    std::cout << "Llama backend freed." << std::endl;
    return 0;
}
