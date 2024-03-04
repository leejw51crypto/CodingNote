# 
## how to install llama.cpp
1. git clone git@github.com:ggerganov/llama.cpp.git
2. cd llama.cpp
3. mkdir build 
4. cd build
5. cmake ..
6. make
7. make install

## installed files
```
-- Installing: /usr/local/include/ggml.h
-- Installing: /usr/local/include/ggml-alloc.h
-- Installing: /usr/local/include/ggml-backend.h
-- Installing: /usr/local/include/ggml-metal.h
-- Installing: /usr/local/lib/libllama.a
-- Installing: /usr/local/include/llama.h
```


## write hello world
- c++ source 
```
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

```
- compile
```
g++ a.cpp -I /usr/local/include -lllama -L /usr/local/lib -framework Accelerate -framework Metal -lobjc -framework Foundation
```
- run
```
./a.out
Initializing llama backend...
Llama backend initialized successfully.
Llama backend freed.
```
