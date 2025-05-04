#!/usr/bin/env python3
import json
import random
import time
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union


# MARK: - Models
@dataclass
class User:
    id: int
    name: str
    email: str
    active: bool
    scores: List[int]
    metadata: Dict[str, str]
    
    @staticmethod
    def generate_fake() -> 'User':
        metadata = {
            "registration_date": "2023-05-15",
            "last_login": "2023-09-20"
        }
        
        return User(
            id=random.randint(1, 10000),
            name=f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            email=f"{random.choice(EMAIL_USERNAMES)}@{random.choice(EMAIL_DOMAINS)}",
            active=random.choice([True, False]),
            scores=[random.randint(1, 100) for _ in range(3)],
            metadata=metadata
        )


@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str
    
    @staticmethod
    def generate_fake() -> 'Address':
        return Address(
            street=f"{random.randint(1, 1000)} Main St",
            city=random.choice(CITIES),
            country=random.choice(COUNTRIES),
            postal_code=f"{random.randint(10000, 99999):05d}"
        )


@dataclass
class UserWithAddress:
    user: User
    address: Address
    created_at: int
    
    @staticmethod
    def generate_fake() -> 'UserWithAddress':
        return UserWithAddress(
            user=User.generate_fake(),
            address=Address.generate_fake(),
            created_at=int(time.time())
        )


# MARK: - Product schema evolution example
@dataclass
class ProductV1:
    id: int
    name: str
    price: float


@dataclass
class ProductV2:
    id: int
    name: str
    price: float
    description: Optional[str] = None
    in_stock: bool = False
    tags: List[str] = field(default_factory=list)


# MARK: - Sample data for random generation
FIRST_NAMES = ["John", "Jane", "Michael", "Sara", "Robert", "Emma", "David", "Olivia"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia"]
EMAIL_USERNAMES = ["user", "person", "contact", "info", "admin", "support", "dev", "sales"]
EMAIL_DOMAINS = ["example.com", "test.org", "mail.com", "domain.net", "service.io"]
CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin"]
COUNTRIES = ["USA", "UK", "Japan", "France", "Germany"]


# MARK: - FlexBuffer simulation
# Note: Python doesn't have a direct equivalent to FlexBuffers, so we're simulating it with JSON
class FlexBuffers:
    @staticmethod
    def to_bytes(value) -> bytes:
        """Serialize an object to bytes (simulating FlexBuffer)"""
        if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
            if isinstance(value, list):
                # Handle lists of objects
                value = [asdict(item) if hasattr(item, '__dataclass_fields__') else item for item in value]
        elif hasattr(value, '__dataclass_fields__'):
            # Convert dataclass to dict
            value = asdict(value)
        elif hasattr(value, '__dict__'):
            # Convert other objects to dict
            value = value.__dict__
        
        return json.dumps(value).encode('utf-8')
    
    @staticmethod
    def from_bytes(data_bytes, cls=None):
        """Deserialize bytes to an object (simulating FlexBuffer)"""
        data = json.loads(data_bytes.decode('utf-8'))
        
        if cls:
            # If a class is specified, convert the dict to an instance of that class
            if hasattr(cls, 'from_dict'):
                return cls.from_dict(data)
            elif hasattr(cls, '__annotations__'):
                return cls(**data)
            else:
                return data
        else:
            return data


# A simple Reader to simulate FlexBuffer's Reader
class Reader:
    def __init__(self, data_bytes):
        self.data = json.loads(data_bytes.decode('utf-8'))
    
    def get_string(self, key) -> str:
        return self.data.get(key, "")
    
    def get_int(self, key) -> int:
        return self.data.get(key, 0)
    
    def get_float(self, key) -> float:
        return self.data.get(key, 0.0)
    
    def get_bool(self, key) -> bool:
        return self.data.get(key, False)
    
    def get_array(self, key) -> list:
        return self.data.get(key, [])
    
    def has_key(self, key) -> bool:
        return key in self.data


# Manual builder to simulate FlexBuffer's Builder
class Builder:
    def __init__(self):
        self.data = {}
    
    def push(self, key, value):
        self.data[key] = value
    
    def push_array(self, key, values):
        self.data[key] = values
    
    def get_bytes(self) -> bytes:
        return json.dumps(self.data).encode('utf-8')


def dict_to_str(d, indent=0):
    """Helper function to pretty print dictionaries for output"""
    indent_str = "    " * indent
    result = []
    
    for k, v in d.items():
        if isinstance(v, dict):
            result.append(f"{indent_str}{k}: {{")
            result.append(dict_to_str(v, indent + 1))
            result.append(f"{indent_str}}}")
        elif isinstance(v, list):
            result.append(f"{indent_str}{k}: [")
            for item in v:
                if isinstance(item, dict):
                    result.append(f"{indent_str}    {{")
                    result.append(dict_to_str(item, indent + 2))
                    result.append(f"{indent_str}    }}")
                else:
                    result.append(f"{indent_str}    {item}")
            result.append(f"{indent_str}]")
        else:
            result.append(f"{indent_str}{k}: {v}")
    
    return "\n".join(result)


# MARK: - Main function
def main():
    print("FlexBuffer Example in Python")
    print("===========================")
    
    # 1. Basic example: Generate fake user data
    user = User.generate_fake()
    print(f"\nOriginal user data: {user}")
    
    # Serialize to FlexBuffer (JSON in our case)
    serialized = FlexBuffers.to_bytes(user)
    print(f"Serialized data size: {len(serialized)} bytes")
    
    # Deserialize from FlexBuffer
    deserialized = FlexBuffers.from_bytes(serialized, User)
    print(f"Deserialized user data: {deserialized}")
    
    # 2. Nested structures example
    users = [
        User.generate_fake(),
        User.generate_fake(),
        User.generate_fake()
    ]
    
    # Serialize the array of users
    serialized_users = FlexBuffers.to_bytes(users)
    print(f"\nSerialized users data size: {len(serialized_users)} bytes")
    
    # Deserialize the array of users
    deserialized_users = [User(**u) for u in FlexBuffers.from_bytes(serialized_users)]
    print(f"Deserialized {len(deserialized_users)} users")
    
    # 3. Manual serialization using the Builder
    builder = Builder()
    builder.push("int_value", 42)
    builder.push("float_value", 3.14159)
    builder.push("string_value", "hello flexbuffers")
    builder.push("bool_value", True)
    builder.push_array("array_value", [1, 2, 3])
    
    manual_serialized = builder.get_bytes()
    
    # Manual deserialization using the Reader
    reader = Reader(manual_serialized)
    print("\nManual deserialization results:")
    print(f"int_value: {reader.get_int('int_value')}")
    print(f"float_value: {reader.get_float('float_value')}")
    print(f"string_value: {reader.get_string('string_value')}")
    print(f"bool_value: {reader.get_bool('bool_value')}")
    
    array = reader.get_array("array_value")
    print(f"array_value: {array}")
    
    # 4. Complex nested structure with custom types
    user_with_address = UserWithAddress.generate_fake()
    print(f"\nComplex user with address: {user_with_address}")
    
    # Serialize to FlexBuffer
    serialized_complex = FlexBuffers.to_bytes(user_with_address)
    print(f"Serialized complex data size: {len(serialized_complex)} bytes")
    
    # Deserialize from FlexBuffer (manual reconstruction for nested types)
    deserialized_dict = FlexBuffers.from_bytes(serialized_complex)
    deserialized_complex = UserWithAddress(
        user=User(**deserialized_dict["user"]),
        address=Address(**deserialized_dict["address"]),
        created_at=deserialized_dict["created_at"]
    )
    print(f"Successfully deserialized complex structure with timestamp: {deserialized_complex.created_at}")
    
    # 5. Binary data example
    binary_data = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE])
    
    # Serialize binary data (using base64 encoding)
    base64_encoded = base64.b64encode(binary_data).decode('utf-8')
    serialized_binary = FlexBuffers.to_bytes(base64_encoded)
    
    # Deserialize binary data
    base64_string = FlexBuffers.from_bytes(serialized_binary)
    binary_vec = base64.b64decode(base64_string)
    
    print(f"\nOriginal binary: {', '.join(['0x{:02X}'.format(b) for b in binary_data])}")
    print(f"Deserialized binary: {', '.join(['0x{:02X}'.format(b) for b in binary_vec])}")
    
    # 6. Data manipulation example
    print("\n== Data manipulation example ==")
    
    # Create a map with initial data
    data_builder = Builder()
    data_builder.push("name", "John Doe")
    data_builder.push("age", 30)
    data_builder.push_array("tags", ["developer", "python"])
    
    initial_data = data_builder.get_bytes()
    
    # Deserialize the data
    data_reader = Reader(initial_data)
    print("Initial data:")
    print(f"Name: {data_reader.get_string('name')}")
    print(f"Age: {data_reader.get_int('age')}")
    print(f"Tags: {data_reader.get_array('tags')}")
    
    # Create a new map with modified data
    modified_builder = Builder()
    modified_builder.push("name", "John Doe")  # Keep existing field
    # Remove "age" field
    
    # Create updated tags
    modified_builder.push_array("tags", ["developer", "python", "flexbuffers"])  # Add new tag
    
    # Add new fields
    modified_builder.push("email", "john@example.com")
    modified_builder.push("active", True)
    
    modified_data = modified_builder.get_bytes()
    
    # Deserialize the modified data
    modified_reader = Reader(modified_data)
    print("\nModified data (added/removed fields):")
    print(f"Name: {modified_reader.get_string('name')}")
    
    # Check for removed field
    print(f"Age present: {modified_reader.has_key('age')}")
    if modified_reader.has_key('age'):
        print(f"Age: {modified_reader.get_int('age')}")
    
    # Check for new fields
    print(f"Email: {modified_reader.get_string('email')}")
    print(f"Active: {modified_reader.get_bool('active')}")
    print(f"Tags: {modified_reader.get_array('tags')}")
    
    # 7. Schema evolution example
    print("\n== Schema evolution example ==")
    
    # Create a product with the original schema
    product_v1 = ProductV1(id=101, name="FlexBuff Widget", price=19.99)
    
    # Serialize using original schema
    serialized_v1 = FlexBuffers.to_bytes(product_v1)
    print(f"ProductV1 serialized size: {len(serialized_v1)} bytes")
    
    # Deserialize using the original schema
    deserialized_v1 = FlexBuffers.from_bytes(serialized_v1, ProductV1)
    print(f"ProductV1 deserialized: id={deserialized_v1.id}, name={deserialized_v1.name}, price={deserialized_v1.price}")
    
    # Now try to deserialize the V1 data using the V2 schema
    v1_dict = FlexBuffers.from_bytes(serialized_v1)
    v2_from_v1 = ProductV2(
        id=v1_dict["id"],
        name=v1_dict["name"],
        price=v1_dict["price"]
    )
    
    print("\nV1 data read as V2 (forward compatibility):")
    print(f"id: {v2_from_v1.id}")
    print(f"name: {v2_from_v1.name}")
    print(f"price: {v2_from_v1.price}")
    print(f"description: {v2_from_v1.description}")
    print(f"in_stock: {v2_from_v1.in_stock} (default)")
    print(f"tags count: {len(v2_from_v1.tags)} (default)")
    
    # Create a product with the new schema
    product_v2 = ProductV2(
        id=102,
        name="FlexBuff Pro",
        price=29.99,
        description="Enhanced FlexBuffer widget with extra features",
        in_stock=True,
        tags=["new", "improved", "featured"]
    )
    
    # Serialize using new schema
    serialized_v2 = FlexBuffers.to_bytes(product_v2)
    print(f"\nProductV2 serialized size: {len(serialized_v2)} bytes")
    
    # Deserialize using the new schema
    deserialized_v2 = FlexBuffers.from_bytes(serialized_v2, ProductV2)
    print("ProductV2 deserialized:")
    print(f"id: {deserialized_v2.id}")
    print(f"name: {deserialized_v2.name}")
    print(f"price: {deserialized_v2.price}")
    print(f"description: {deserialized_v2.description}")
    print(f"in_stock: {deserialized_v2.in_stock}")
    print(f"tags: {deserialized_v2.tags}")
    
    # Now try to deserialize the V2 data using the V1 schema
    v2_dict = FlexBuffers.from_bytes(serialized_v2)
    v1_from_v2 = ProductV1(
        id=v2_dict["id"],
        name=v2_dict["name"],
        price=v2_dict["price"]
    )
    
    print("\nV2 data read as V1 (backward compatibility):")
    print(f"id: {v1_from_v2.id}")
    print(f"name: {v1_from_v2.name}")
    print(f"price: {v1_from_v2.price}")
    print("(Additional V2 fields are ignored when reading as V1)")


if __name__ == "__main__":
    main() 