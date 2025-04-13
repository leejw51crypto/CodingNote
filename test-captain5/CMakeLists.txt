cmake_minimum_required(VERSION 3.15)
project(capnp_test_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Cap'n Proto
find_package(CapnProto REQUIRED)

# Create directory for generated files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto)

# Generate C++ files from schema
capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/proto/book.capnp)

# Create the C wrapper library as a static library
add_library(book_wrapper STATIC
    Sources/CCapnp/book_wrapper.cpp
    ${CAPNP_SRCS}
)

target_include_directories(book_wrapper PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CAPNP_INCLUDE_DIRS}
    Sources/CCapnp/include
)

target_link_libraries(book_wrapper PUBLIC
    CapnProto::capnp
)

# Set output directory for the static library
set_target_properties(book_wrapper PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Create the C++ example executable
add_executable(cpp_main 
    cpp/main.cpp
    ${CAPNP_SRCS}
)

target_include_directories(cpp_main PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CAPNP_INCLUDE_DIRS}
)

target_link_libraries(cpp_main PRIVATE
    CapnProto::capnp
)

# Install the library and headers
install(TARGETS book_wrapper
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES 
    Sources/CCapnp/include/book_wrapper.h
    ${CAPNP_HDRS}
    DESTINATION include
) 