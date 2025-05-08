import capnp
import os

# Load the Cap'n Proto schema
capnp.remove_import_hook()
hello_capnp = capnp.load(os.path.join(os.path.dirname(__file__), "hello.capnp"))


def encode_hello_world(message: str) -> bytes:
    msg = hello_capnp.HelloWorld.new_message()
    msg.message = message
    return msg.to_bytes()


def decode_hello_world(data: bytes) -> str:
    with hello_capnp.HelloWorld.from_bytes(data) as msg:
        return msg.message


if __name__ == "__main__":
    # Example usage
    original_message = "Hello, world!"
    encoded = encode_hello_world(original_message)
    print(f"Encoded bytes: {encoded}")
    decoded = decode_hello_world(encoded)
    print(f"Decoded message: {decoded}")
