#include <iostream>
#include <memory>
#include <cstring>

extern "C" {
    void* create_channel();
    const char* receive_message(void* channel_ptr, int milliseconds);
    void destroy_channel(void* channel_ptr);
}
#include "./target/cxxbridge/mylib/src/lib.rs.h"
using namespace std;
using namespace mysdk;
int main() {
	 void* channel_ptr = create_channel();

    while(true) {
        const char* message = receive_message(channel_ptr,1000);
        if (message != nullptr) {
            std::cout << "Received message from Rust: " << message << std::endl;
            std::free((void*) message);
        }
		else {
			std::cout<<"waiting"<<std::endl;
		}
    }

    destroy_channel(channel_ptr);
	return 0;
}
