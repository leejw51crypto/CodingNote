#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <ctime>
#include <random>
#include <memory>
#include <sstream>
#include <iomanip>
#include <optional>
#include <algorithm>

// FlexBuffers library - included inline for simplicity
// In a real application, you would use the actual FlexBuffers library from FlatBuffers
#include "flexbuffers.h"

// Random utility functions
class Random {
private:
    static std::mt19937 generator;

public:
    static void seed() {
        generator.seed(static_cast<unsigned int>(time(nullptr)));
    }

    static int getInt(int min, int max) {
        std::uniform_int_distribution<int> distribution(min, max);
        return distribution(generator);
    }

    static bool getBool() {
        return getInt(0, 1) == 1;
    }

    template<typename T>
    static T getChoice(const std::vector<T>& choices) {
        if (choices.empty()) return T();
        return choices[getInt(0, static_cast<int>(choices.size()) - 1)];
    }
};

std::mt19937 Random::generator;

// Sample data for random generation
const std::vector<std::string> FIRST_NAMES = {"John", "Jane", "Michael", "Sara", "Robert", "Emma", "David", "Olivia"};
const std::vector<std::string> LAST_NAMES = {"Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia"};
const std::vector<std::string> EMAIL_USERNAMES = {"user", "person", "contact", "info", "admin", "support", "dev", "sales"};
const std::vector<std::string> EMAIL_DOMAINS = {"example.com", "test.org", "mail.com", "domain.net", "service.io"};
const std::vector<std::string> CITIES = {"New York", "London", "Tokyo", "Paris", "Berlin"};
const std::vector<std::string> COUNTRIES = {"USA", "UK", "Japan", "France", "Germany"};

// MARK: - Models
class User {
public:
    uint32_t id;
    std::string name;
    std::string email;
    bool active;
    std::vector<int32_t> scores;
    std::unordered_map<std::string, std::string> metadata;

    // Constructor
    User(uint32_t id, const std::string& name, const std::string& email, bool active, 
         const std::vector<int32_t>& scores, const std::unordered_map<std::string, std::string>& metadata)
        : id(id), name(name), email(email), active(active), scores(scores), metadata(metadata) {}

    // Serialization
    void serialize(flexbuffers::Builder& builder) const {
        auto map = builder.StartMap();
        builder.Int("id", id);
        builder.String("name", name);
        builder.String("email", email);
        builder.Bool("active", active);
        
        auto vec = builder.StartVector("scores");
        for (int32_t score : scores) {
            builder.Int(score);
        }
        builder.EndVector(vec, false, false);
        
        auto metaMap = builder.StartMap("metadata");
        for (const auto& [key, value] : metadata) {
            builder.String(key.c_str(), value);
        }
        builder.EndMap(metaMap);
        
        builder.EndMap(map);
    }

    // Deserialization
    static User deserialize(const flexbuffers::Map& map) {
        uint32_t id = map["id"].AsUInt32();
        std::string name = map["name"].AsString().str();
        std::string email = map["email"].AsString().str();
        bool active = map["active"].AsBool();
        
        auto scores_vec = map["scores"].AsVector();
        std::vector<int32_t> scores;
        for (size_t i = 0; i < scores_vec.size(); i++) {
            scores.push_back(scores_vec[i].AsInt32());
        }
        
        auto meta_map = map["metadata"].AsMap();
        std::unordered_map<std::string, std::string> metadata;
        auto keys = meta_map.Keys();
        for (size_t i = 0; i < meta_map.size(); i++) {
            std::string key = keys[i].str();
            metadata[key] = meta_map[key].AsString().str();
        }
        
        return User(id, name, email, active, scores, metadata);
    }

    // Generate fake user
    static User generateFake() {
        std::unordered_map<std::string, std::string> metadata;
        metadata["registration_date"] = "2023-05-15";
        metadata["last_login"] = "2023-09-20";
        
        return User(
            Random::getInt(1, 10000),
            Random::getChoice(FIRST_NAMES) + " " + Random::getChoice(LAST_NAMES),
            Random::getChoice(EMAIL_USERNAMES) + "@" + Random::getChoice(EMAIL_DOMAINS),
            Random::getBool(),
            {Random::getInt(1, 100), Random::getInt(1, 100), Random::getInt(1, 100)},
            metadata
        );
    }

    // Debug string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "User(id=" << id << ", name=" << name << ", email=" << email << ", active=" << std::boolalpha << active;
        
        ss << ", scores=[";
        for (size_t i = 0; i < scores.size(); i++) {
            ss << scores[i];
            if (i < scores.size() - 1) ss << ", ";
        }
        ss << "]";
        
        ss << ", metadata={";
        size_t i = 0;
        for (const auto& [key, value] : metadata) {
            ss << key << ": " << value;
            if (i < metadata.size() - 1) ss << ", ";
            i++;
        }
        ss << "})";
        
        return ss.str();
    }
};

class Address {
public:
    std::string street;
    std::string city;
    std::string country;
    std::string postalCode;

    // Constructor
    Address(const std::string& street, const std::string& city, const std::string& country, const std::string& postalCode)
        : street(street), city(city), country(country), postalCode(postalCode) {}

    // Serialization
    void serialize(flexbuffers::Builder& builder) const {
        auto map = builder.StartMap();
        builder.String("street", street);
        builder.String("city", city);
        builder.String("country", country);
        builder.String("postalCode", postalCode);
        builder.EndMap(map);
    }

    // Deserialization
    static Address deserialize(const flexbuffers::Map& map) {
        std::string street = map["street"].AsString().str();
        std::string city = map["city"].AsString().str();
        std::string country = map["country"].AsString().str();
        std::string postalCode = map["postalCode"].AsString().str();
        
        return Address(street, city, country, postalCode);
    }

    // Generate fake address
    static Address generateFake() {
        return Address(
            std::to_string(Random::getInt(1, 1000)) + " Main St",
            Random::getChoice(CITIES),
            Random::getChoice(COUNTRIES),
            std::to_string(Random::getInt(10000, 99999))
        );
    }

    // Debug string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "Address(street=" << street << ", city=" << city
           << ", country=" << country << ", postalCode=" << postalCode << ")";
        return ss.str();
    }
};

class UserWithAddress {
public:
    User user;
    Address address;
    uint64_t createdAt;

    // Constructor
    UserWithAddress(const User& user, const Address& address, uint64_t createdAt)
        : user(user), address(address), createdAt(createdAt) {}

    // Serialization
    void serialize(flexbuffers::Builder& builder) const {
        auto map = builder.StartMap();
        
        auto userMap = builder.StartMap("user");
        user.serialize(builder);
        builder.EndMap(userMap);
        
        auto addrMap = builder.StartMap("address");
        address.serialize(builder);
        builder.EndMap(addrMap);
        
        builder.UInt("createdAt", createdAt);
        
        builder.EndMap(map);
    }

    // Deserialization
    static UserWithAddress deserialize(const flexbuffers::Map& map) {
        auto userMap = map["user"].AsMap();
        auto addressMap = map["address"].AsMap();
        uint64_t createdAt = map["createdAt"].AsUInt64();
        
        return UserWithAddress(
            User::deserialize(userMap),
            Address::deserialize(addressMap),
            createdAt
        );
    }

    // Generate fake user with address
    static UserWithAddress generateFake() {
        return UserWithAddress(
            User::generateFake(),
            Address::generateFake(),
            static_cast<uint64_t>(time(nullptr))
        );
    }

    // Debug string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "UserWithAddress(user=" << user.toString() 
           << ", address=" << address.toString()
           << ", createdAt=" << createdAt << ")";
        return ss.str();
    }
};

// MARK: - Product schema evolution example
class ProductV1 {
public:
    uint32_t id;
    std::string name;
    double price;

    // Constructor
    ProductV1(uint32_t id, const std::string& name, double price)
        : id(id), name(name), price(price) {}

    // Serialization
    void serialize(flexbuffers::Builder& builder) const {
        auto map = builder.StartMap();
        builder.UInt("id", id);
        builder.String("name", name);
        builder.Double("price", price);
        builder.EndMap(map);
    }

    // Deserialization
    static ProductV1 deserialize(const flexbuffers::Map& map) {
        uint32_t id = map["id"].AsUInt32();
        std::string name = map["name"].AsString().str();
        double price = map["price"].AsDouble();
        
        return ProductV1(id, name, price);
    }

    // Debug string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "ProductV1(id=" << id << ", name=" << name << ", price=" << std::fixed << std::setprecision(2) << price << ")";
        return ss.str();
    }
};

class ProductV2 {
public:
    uint32_t id;
    std::string name;
    double price;
    std::optional<std::string> description;
    bool inStock;
    std::vector<std::string> tags;

    // Constructor
    ProductV2(uint32_t id, const std::string& name, double price,
              const std::optional<std::string>& description = std::nullopt,
              bool inStock = false, const std::vector<std::string>& tags = {})
        : id(id), name(name), price(price), description(description), inStock(inStock), tags(tags) {}

    // Serialization
    void serialize(flexbuffers::Builder& builder) const {
        auto map = builder.StartMap();
        builder.UInt("id", id);
        builder.String("name", name);
        builder.Double("price", price);
        
        if (description.has_value()) {
            builder.String("description", description.value());
        } else {
            builder.Null("description");
        }
        
        builder.Bool("inStock", inStock);
        
        auto tagsVec = builder.StartVector("tags");
        for (const auto& tag : tags) {
            builder.String(tag);
        }
        builder.EndVector(tagsVec, false, false);
        
        builder.EndMap(map);
    }

    // Deserialization
    static ProductV2 deserialize(const flexbuffers::Map& map) {
        uint32_t id = map["id"].AsUInt32();
        std::string name = map["name"].AsString().str();
        double price = map["price"].AsDouble();
        
        std::optional<std::string> description;
        if (!map["description"].IsNull()) {
            description = map["description"].AsString().str();
        }
        
        bool inStock = false;
        if (map.Find("inStock") != nullptr) {
            inStock = map["inStock"].AsBool();
        }
        
        std::vector<std::string> tags;
        if (map.Find("tags") != nullptr) {
            auto tags_vec = map["tags"].AsVector();
            for (size_t i = 0; i < tags_vec.size(); i++) {
                tags.push_back(tags_vec[i].AsString().str());
            }
        }
        
        return ProductV2(id, name, price, description, inStock, tags);
    }

    // Debug string representation
    std::string toString() const {
        std::stringstream ss;
        ss << "ProductV2(id=" << id << ", name=" << name 
           << ", price=" << std::fixed << std::setprecision(2) << price
           << ", description=" << (description.has_value() ? description.value() : "null")
           << ", inStock=" << std::boolalpha << inStock
           << ", tags=[";
        
        for (size_t i = 0; i < tags.size(); i++) {
            ss << tags[i];
            if (i < tags.size() - 1) ss << ", ";
        }
        ss << "])";
        
        return ss.str();
    }
};

// Utility to print binary data
std::string formatBinary(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    for (size_t i = 0; i < data.size(); i++) {
        ss << "0x" << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(data[i]);
        if (i < data.size() - 1) ss << ", ";
    }
    return ss.str();
}

// Main function
int main() {
    Random::seed();
    
    std::cout << "FlexBuffer Example in C++" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // 1. Basic example: Generate fake user data
    User user = User::generateFake();
    std::cout << "\nOriginal user data: " << user.toString() << std::endl;
    
    // Serialize to FlexBuffer
    flexbuffers::Builder builder;
    builder.Clear();
    user.serialize(builder);
    builder.Finish();
    
    auto serialized = builder.GetBuffer();
    std::cout << "Serialized data size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize from FlexBuffer
    auto root = flexbuffers::GetRoot(serialized);
    auto deserialized = User::deserialize(root.AsMap());
    std::cout << "Deserialized user data: " << deserialized.toString() << std::endl;
    
    // 2. Nested structures example
    std::vector<User> users = {
        User::generateFake(),
        User::generateFake(),
        User::generateFake()
    };
    
    // Serialize the vector of users
    builder.Clear();
    auto usersVec = builder.StartVector();
    for (const auto& u : users) {
        auto userMap = builder.StartMap();
        u.serialize(builder);
        builder.EndMap(userMap);
    }
    builder.EndVector(usersVec, false, false);
    builder.Finish();
    
    auto serialized_users = builder.GetBuffer();
    std::cout << "\nSerialized users data size: " << serialized_users.size() << " bytes" << std::endl;
    
    // Deserialize the vector of users
    auto users_root = flexbuffers::GetRoot(serialized_users);
    auto users_vec = users_root.AsVector();
    std::cout << "Deserialized " << users_vec.size() << " users" << std::endl;
    
    // 3. Manual serialization using the Builder
    builder.Clear();
    auto manualMap = builder.StartMap();
    builder.Int("int_value", 42);
    builder.Double("float_value", 3.14159);
    builder.String("string_value", "hello flexbuffers");
    builder.Bool("bool_value", true);
    
    auto arrayValue = builder.StartVector("array_value");
    builder.Int(1);
    builder.Int(2);
    builder.Int(3);
    builder.EndVector(arrayValue, false, false);
    
    builder.EndMap(manualMap);
    builder.Finish();
    
    auto manual_serialized = builder.GetBuffer();
    
    // Manual deserialization
    auto manual_root = flexbuffers::GetRoot(manual_serialized);
    auto manual_map = manual_root.AsMap();
    
    std::cout << "\nManual deserialization results:" << std::endl;
    std::cout << "int_value: " << manual_map["int_value"].AsInt32() << std::endl;
    std::cout << "float_value: " << manual_map["float_value"].AsDouble() << std::endl;
    std::cout << "string_value: " << manual_map["string_value"].AsString().str() << std::endl;
    std::cout << "bool_value: " << std::boolalpha << manual_map["bool_value"].AsBool() << std::endl;
    
    auto array = manual_map["array_value"].AsVector();
    std::cout << "array_value: [";
    for (size_t i = 0; i < array.size(); i++) {
        std::cout << array[i].AsInt32();
        if (i < array.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 4. Complex nested structure with custom types
    UserWithAddress userWithAddress = UserWithAddress::generateFake();
    std::cout << "\nComplex user with address: " << userWithAddress.toString() << std::endl;
    
    // Serialize to FlexBuffer
    builder.Clear();
    userWithAddress.serialize(builder);
    builder.Finish();
    
    auto serialized_complex = builder.GetBuffer();
    std::cout << "Serialized complex data size: " << serialized_complex.size() << " bytes" << std::endl;
    
    // Deserialize from FlexBuffer
    auto complex_root = flexbuffers::GetRoot(serialized_complex);
    auto deserialized_complex = UserWithAddress::deserialize(complex_root.AsMap());
    std::cout << "Successfully deserialized complex structure with timestamp: " 
              << deserialized_complex.createdAt << std::endl;
    
    // 5. Binary data example
    std::vector<uint8_t> binary_data = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    
    // Serialize binary data
    builder.Clear();
    auto binary_vec = builder.StartVector();
    for (auto b : binary_data) {
        builder.UInt(b);
    }
    builder.EndVector(binary_vec, false, false);
    builder.Finish();
    
    auto serialized_binary = builder.GetBuffer();
    // Mark this as binary data
    if (serialized_binary.size() > 0) {
        serialized_binary[0] = 2;
    }
    
    // Deserialize binary data
    auto binary_root = flexbuffers::GetRoot(serialized_binary);
    auto binary_array = binary_root.AsVector();
    
    std::vector<uint8_t> binary_deserialized;
    for (size_t i = 0; i < binary_array.size(); i++) {
        binary_deserialized.push_back(static_cast<uint8_t>(binary_array[i].AsUInt8()));
    }
    
    std::cout << "\nOriginal binary: [" << formatBinary(binary_data) << "]" << std::endl;
    std::cout << "Deserialized binary: [" << formatBinary(binary_deserialized) << "]" << std::endl;
    
    // 6. Data manipulation example
    std::cout << "\n== Data manipulation example ==" << std::endl;
    
    // Create a map with initial data
    builder.Clear();
    auto dataMap = builder.StartMap();
    builder.String("name", "John Doe");
    builder.Int("age", 30);
    
    auto tagsVec = builder.StartVector("tags");
    builder.String("developer");
    builder.String("c++");
    builder.EndVector(tagsVec, false, false);
    
    builder.EndMap(dataMap);
    builder.Finish();
    
    auto initial_data = builder.GetBuffer();
    // Mark this as initial data map
    if (initial_data.size() > 0) {
        initial_data[0] = 3;
    }
    
    // Read the initial data
    auto initial_root = flexbuffers::GetRoot(initial_data);
    auto initial_map = initial_root.AsMap();
    
    std::cout << "Initial data:" << std::endl;
    std::cout << "Name: " << initial_map["name"].AsString().str() << std::endl;
    std::cout << "Age: " << initial_map["age"].AsInt32() << std::endl;
    
    auto initial_tags = initial_map["tags"].AsVector();
    std::cout << "Tags: [";
    for (size_t i = 0; i < initial_tags.size(); i++) {
        std::cout << initial_tags[i].AsString().str();
        if (i < initial_tags.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Create a new map with modified data
    builder.Clear();
    auto modifiedMap = builder.StartMap();
    builder.String("name", "John Doe");  // Keep existing field
    // Remove "age" field
    
    // Create updated tags with an additional tag
    auto modifiedTags = builder.StartVector("tags");
    builder.String("developer");
    builder.String("c++");
    builder.String("flexbuffers");  // Add a new tag
    builder.EndVector(modifiedTags, false, false);
    
    // Add new fields
    builder.String("email", "john@example.com");
    builder.Bool("active", true);
    
    builder.EndMap(modifiedMap);
    builder.Finish();
    
    auto modified_data = builder.GetBuffer();
    // Mark this as modified data map
    if (modified_data.size() > 0) {
        modified_data[0] = 4;
    }
    
    // Read the modified data
    auto modified_root = flexbuffers::GetRoot(modified_data);
    auto modified_map = modified_root.AsMap();
    
    std::cout << "\nModified data (added/removed fields):" << std::endl;
    std::cout << "Name: " << modified_map["name"].AsString().str() << std::endl;
    
    // Check for removed field
    bool age_present = modified_map.Find("age") != nullptr;
    std::cout << "Age present: " << std::boolalpha << age_present << std::endl;
    
    // Check for new fields
    std::cout << "Email: " << modified_map["email"].AsString().str() << std::endl;
    std::cout << "Active: " << std::boolalpha << modified_map["active"].AsBool() << std::endl;
    
    auto modified_tags = modified_map["tags"].AsVector();
    std::cout << "Tags: [";
    for (size_t i = 0; i < modified_tags.size(); i++) {
        std::cout << modified_tags[i].AsString().str();
        if (i < modified_tags.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 7. Schema evolution example
    std::cout << "\n== Schema evolution example ==" << std::endl;
    
    // Create a product with the original schema
    ProductV1 productV1(101, "FlexBuff Widget", 19.99);
    
    // Serialize using original schema
    builder.Clear();
    productV1.serialize(builder);
    builder.Finish();
    
    auto serialized_v1 = builder.GetBuffer();
    std::cout << "ProductV1 serialized size: " << serialized_v1.size() << " bytes" << std::endl;
    
    // Deserialize using the original schema
    auto v1_root = flexbuffers::GetRoot(serialized_v1);
    auto deserialized_v1 = ProductV1::deserialize(v1_root.AsMap());
    std::cout << "ProductV1 deserialized: " << deserialized_v1.toString() << std::endl;
    
    // Now try to deserialize the V1 data using the V2 schema - demonstrating forward compatibility
    auto v1_map = v1_root.AsMap();
    ProductV2 v2_from_v1(
        v1_map["id"].AsUInt32(),
        v1_map["name"].AsString().str(),
        v1_map["price"].AsDouble()
    );
    
    std::cout << "\nV1 data read as V2 (forward compatibility):" << std::endl;
    std::cout << v2_from_v1.toString() << std::endl;
    
    // Create a product with the new schema
    ProductV2 productV2(
        102, 
        "FlexBuff Pro", 
        29.99, 
        "Enhanced FlexBuffer widget with extra features", 
        true, 
        {"new", "improved", "featured"}
    );
    
    // Serialize using new schema
    builder.Clear();
    productV2.serialize(builder);
    builder.Finish();
    
    auto serialized_v2 = builder.GetBuffer();
    std::cout << "\nProductV2 serialized size: " << serialized_v2.size() << " bytes" << std::endl;
    
    // Deserialize using the new schema
    auto v2_root = flexbuffers::GetRoot(serialized_v2);
    auto deserialized_v2 = ProductV2::deserialize(v2_root.AsMap());
    std::cout << "ProductV2 deserialized: " << deserialized_v2.toString() << std::endl;
    
    // Now try to deserialize the V2 data using the V1 schema - demonstrating backward compatibility
    auto v2_map = v2_root.AsMap();
    ProductV1 v1_from_v2(
        v2_map["id"].AsUInt32(),
        v2_map["name"].AsString().str(),
        v2_map["price"].AsDouble()
    );
    
    std::cout << "\nV2 data read as V1 (backward compatibility):" << std::endl;
    std::cout << v1_from_v2.toString() << std::endl;
    std::cout << "(Additional V2 fields are ignored when reading as V1)" << std::endl;
    
    return 0;
} 