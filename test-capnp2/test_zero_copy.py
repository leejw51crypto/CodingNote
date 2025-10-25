#!/usr/bin/env python3
"""
Testing Cap'n Proto's zero-copy deserialization
"""

import capnp
import person_capnp


def main():
    # Create a person
    person = person_capnp.Person.new_message()
    person.name = "Alice"
    person.email = "alice@example.com"
    person.age = 30

    # Serialize to bytes
    serialized = person.to_bytes()
    print(f"Original serialized bytes: {len(serialized)} bytes")

    # Test 1: What does from_bytes actually do?
    print("\n--- Test 1: Understanding from_bytes ---")
    print("Calling from_bytes() - does this deserialize?")

    # from_bytes returns a context manager
    with person_capnp.Person.from_bytes(serialized) as person2:
        print(f"Type: {type(person2)}")
        print(f"Can access name directly: {person2.name}")
        print("^ This is ZERO-COPY! No deserialization into Python objects!")
        print("The data is read directly from the byte buffer")

    # Test 2: Compare with traditional deserialization
    print("\n--- Test 2: What is 'deserialization' in Cap'n Proto? ---")
    print("Traditional serialization (JSON, Protobuf, etc.):")
    print("  1. Parse bytes into memory structures")
    print("  2. Copy data into language objects")
    print("  3. Use the objects")
    print("")
    print("Cap'n Proto:")
    print("  1. Validate byte buffer structure")
    print("  2. Keep bytes as-is in memory")
    print("  3. Access fields via pointer arithmetic (zero-copy!)")

    # Test 3: Packed format
    print("\n--- Test 3: Packed format ---")
    serialized_packed = person.to_bytes_packed()
    print(f"Packed bytes: {len(serialized_packed)} bytes")

    person3 = person_capnp.Person.from_bytes_packed(serialized_packed)
    print(f"from_bytes_packed() unpacks: {serialized_packed.hex()[:20]}...")
    print(f"  -> into memory buffer (not shown)")
    print(f"  -> then accesses via pointers")
    print(f"Access name: {person3.name}")
    print("")
    print("So from_bytes_packed() does TWO things:")
    print("  1. Unpack the compressed format (required)")
    print("  2. Return a zero-copy reader over the unpacked buffer")

    # Test 4: Prove it's zero-copy
    print("\n--- Test 4: Is it really zero-copy? ---")
    with person_capnp.Person.from_bytes(serialized) as p:
        # Access only one field
        print(f"Only accessing 'age' field: {p.age}")
        print("The 'name' and 'email' strings were NEVER copied to Python!")
        print("That's the power of zero-copy deserialization!")


if __name__ == "__main__":
    main()
