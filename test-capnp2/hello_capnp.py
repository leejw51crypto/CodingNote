#!/usr/bin/env python3
"""
Cap'n Proto Hello World Example
"""

import capnp
import person_capnp


def main():
    # Create a new Person message
    person = person_capnp.Person.new_message()
    person.name = "Alice"
    person.email = "alice@example.com"
    person.age = 30

    print("Created Person:")
    print(f"  Name: {person.name}")
    print(f"  Email: {person.email}")
    print(f"  Age: {person.age}")

    # Serialize to bytes (unpacked)
    serialized = person.to_bytes()
    print(f"\nSerialized (unpacked):")
    print(f"  Length: {len(serialized)} bytes")
    print(f"  Hex: {serialized.hex()}")

    # Clear write flag to avoid warning when serializing again
    person.clear_write_flag()

    # Serialize to bytes (packed/compressed)
    serialized_packed = person.to_bytes_packed()
    print(f"\nSerialized (packed/compressed):")
    print(f"  Length: {len(serialized_packed)} bytes")
    print(f"  Hex: {serialized_packed.hex()}")
    print(f"  Compression ratio: {len(serialized_packed)/len(serialized)*100:.1f}%")

    # Deserialize from unpacked bytes
    with person_capnp.Person.from_bytes(serialized) as person2:
        print("\nDeserialized from unpacked:")
        print(f"  Name: {person2.name}")
        print(f"  Email: {person2.email}")
        print(f"  Age: {person2.age}")

    # Deserialize from packed bytes
    person3 = person_capnp.Person.from_bytes_packed(serialized_packed)
    print("\nDeserialized from packed:")
    print(f"  Name: {person3.name}")
    print(f"  Email: {person3.email}")
    print(f"  Age: {person3.age}")

    # Test: Can we access members directly without explicit unpacking?
    print("\n--- Testing direct member access from packed bytes ---")
    person4 = person_capnp.Person.from_bytes_packed(serialized_packed)
    print("Accessing individual members directly:")
    print(f"  Just name: {person4.name}")
    print(f"  Just email: {person4.email}")
    print(f"  Just age: {person4.age}")
    print("✓ Yes! Members are directly accessible from packed format")

    # Verify they're equal
    print(
        f"\nData matches: {person.name == person3.name and person.email == person3.email and person.age == person3.age}"
    )


if __name__ == "__main__":
    main()
