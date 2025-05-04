import Foundation

// MARK: - Models
struct User: Codable, CustomStringConvertible {
    let id: UInt32
    let name: String
    let email: String
    let active: Bool
    let scores: [Int32]
    let metadata: [String: String]
    
    var description: String {
        return """
        User(
            id: \(id),
            name: \(name),
            email: \(email),
            active: \(active),
            scores: \(scores),
            metadata: \(metadata)
        )
        """
    }
    
    static func generateFake() -> User {
        var metadata: [String: String] = [:]
        metadata["registration_date"] = "2023-05-15"
        metadata["last_login"] = "2023-09-20"
        
        return User(
            id: UInt32.random(in: 1...10000),
            name: "\(firstNames.randomElement()!) \(lastNames.randomElement()!)",
            email: "\(emailUsernames.randomElement()!)@\(emailDomains.randomElement()!)",
            active: Bool.random(),
            scores: [Int32.random(in: 1...100), Int32.random(in: 1...100), Int32.random(in: 1...100)],
            metadata: metadata
        )
    }
}

struct Address: Codable, CustomStringConvertible {
    let street: String
    let city: String
    let country: String
    let postalCode: String
    
    var description: String {
        return """
        Address(
            street: \(street),
            city: \(city),
            country: \(country),
            postalCode: \(postalCode)
        )
        """
    }
    
    static func generateFake() -> Address {
        return Address(
            street: "\(Int.random(in: 1...1000)) Main St",
            city: cities.randomElement()!,
            country: countries.randomElement()!,
            postalCode: String(format: "%05d", Int.random(in: 10000...99999))
        )
    }
}

struct UserWithAddress: Codable, CustomStringConvertible {
    let user: User
    let address: Address
    let createdAt: UInt64
    
    var description: String {
        return """
        UserWithAddress(
            user: \(user),
            address: \(address),
            createdAt: \(createdAt)
        )
        """
    }
    
    static func generateFake() -> UserWithAddress {
        return UserWithAddress(
            user: User.generateFake(),
            address: Address.generateFake(),
            createdAt: UInt64(Date().timeIntervalSince1970)
        )
    }
}

// MARK: - Product schema evolution example
struct ProductV1: Codable, CustomStringConvertible {
    let id: UInt32
    let name: String
    let price: Double
    
    var description: String {
        return "ProductV1(id: \(id), name: \(name), price: \(price))"
    }
}

struct ProductV2: Codable, CustomStringConvertible {
    let id: UInt32
    let name: String
    let price: Double
    let productDescription: String?
    let inStock: Bool
    let tags: [String]
    
    var description: String {
        return """
        ProductV2(
            id: \(id),
            name: \(name),
            price: \(price),
            description: \(productDescription ?? "nil"),
            inStock: \(inStock),
            tags: \(tags)
        )
        """
    }
    
    // Default initializer for V1 to V2 conversion
    init(id: UInt32, name: String, price: Double, productDescription: String? = nil, inStock: Bool = false, tags: [String] = []) {
        self.id = id
        self.name = name
        self.price = price
        self.productDescription = productDescription
        self.inStock = inStock
        self.tags = tags
    }
}

// MARK: - Sample data for random generation
let firstNames = ["John", "Jane", "Michael", "Sara", "Robert", "Emma", "David", "Olivia"]
let lastNames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia"]
let emailUsernames = ["user", "person", "contact", "info", "admin", "support", "dev", "sales"]
let emailDomains = ["example.com", "test.org", "mail.com", "domain.net", "service.io"]
let cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]
let countries = ["USA", "UK", "Japan", "France", "Germany"]

// MARK: - FlexBuffer simulation
// Note: Swift doesn't have a direct equivalent to FlexBuffers, so we're simulating it with JSON
class FlexBuffers {
    static func toData<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        return try encoder.encode(value)
    }
    
    static func fromData<T: Decodable>(_ data: Data) throws -> T {
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }
}

// A simple Reader to simulate FlexBuffer's Reader
class Reader {
    private let json: [String: Any]
    
    init(data: Data) throws {
        if let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            self.json = jsonObject
        } else {
            self.json = [:]
        }
    }
    
    func getString(_ key: String) -> String {
        return json[key] as? String ?? ""
    }
    
    func getInt(_ key: String) -> Int {
        return json[key] as? Int ?? 0
    }
    
    func getDouble(_ key: String) -> Double {
        return json[key] as? Double ?? 0.0
    }
    
    func getBool(_ key: String) -> Bool {
        return json[key] as? Bool ?? false
    }
    
    func getArray(_ key: String) -> [Any] {
        return json[key] as? [Any] ?? []
    }
    
    func hasKey(_ key: String) -> Bool {
        return json[key] != nil
    }
}

// Manual builder to simulate FlexBuffer's Builder
class Builder {
    private var data: [String: Any] = [:]
    
    func push(_ key: String, _ value: Any) {
        data[key] = value
    }
    
    func pushArray(_ key: String, _ values: [Any]) {
        data[key] = values
    }
    
    func getData() throws -> Data {
        return try JSONSerialization.data(withJSONObject: data)
    }
}

// MARK: - Main function equivalent
func main() {
    print("FlexBuffer Example in Swift")
    print("==========================")
    
    do {
        // 1. Basic example: Generate fake user data
        let user = User.generateFake()
        print("\nOriginal user data: \(user)")
        
        // Serialize to FlexBuffer (JSON in our case)
        let serialized = try FlexBuffers.toData(user)
        print("Serialized data size: \(serialized.count) bytes")
        
        // Deserialize from FlexBuffer
        let deserialized: User = try FlexBuffers.fromData(serialized)
        print("Deserialized user data: \(deserialized)")
        
        // 2. Nested structures example
        let users = [
            User.generateFake(),
            User.generateFake(),
            User.generateFake()
        ]
        
        // Serialize the array of users
        let serializedUsers = try FlexBuffers.toData(users)
        print("\nSerialized users data size: \(serializedUsers.count) bytes")
        
        // Deserialize the array of users
        let deserializedUsers: [User] = try FlexBuffers.fromData(serializedUsers)
        print("Deserialized \(deserializedUsers.count) users")
        
        // 3. Manual serialization using the Builder
        let builder = Builder()
        builder.push("int_value", 42)
        builder.push("float_value", 3.14159)
        builder.push("string_value", "hello flexbuffers")
        builder.push("bool_value", true)
        builder.pushArray("array_value", [1, 2, 3])
        
        let manualSerialized = try builder.getData()
        
        // Manual deserialization using the Reader
        let reader = try Reader(data: manualSerialized)
        print("\nManual deserialization results:")
        print("int_value: \(reader.getInt("int_value"))")
        print("float_value: \(reader.getDouble("float_value"))")
        print("string_value: \(reader.getString("string_value"))")
        print("bool_value: \(reader.getBool("bool_value"))")
        
        let array = reader.getArray("array_value")
        print("array_value: \(array)")
        
        // 4. Complex nested structure with custom types
        let userWithAddress = UserWithAddress.generateFake()
        print("\nComplex user with address: \(userWithAddress)")
        
        // Serialize to FlexBuffer
        let serializedComplex = try FlexBuffers.toData(userWithAddress)
        print("Serialized complex data size: \(serializedComplex.count) bytes")
        
        // Deserialize from FlexBuffer
        let deserializedComplex: UserWithAddress = try FlexBuffers.fromData(serializedComplex)
        print("Successfully deserialized complex structure with timestamp: \(deserializedComplex.createdAt)")
        
        // 5. Binary data example
        let binaryData: [UInt8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE]
        
        // Serialize binary data (using Data wrapper)
        let binaryDataWrapper = Data(binaryData)
        let serializedBinary = try JSONEncoder().encode(binaryDataWrapper.base64EncodedString())
        
        // Deserialize binary data
        let base64String = try JSONDecoder().decode(String.self, from: serializedBinary)
        let binaryVec = Data(base64Encoded: base64String)!
        
        print("\nOriginal binary: \(binaryData.map { String(format: "0x%02X", $0) }.joined(separator: ", "))")
        print("Deserialized binary: \(binaryVec.map { String(format: "0x%02X", $0) }.joined(separator: ", "))")
        
        // 6. Data manipulation example
        print("\n== Data manipulation example ==")
        
        // Create a map with initial data
        let dataBuilder = Builder()
        dataBuilder.push("name", "John Doe")
        dataBuilder.push("age", 30)
        dataBuilder.pushArray("tags", ["developer", "swift"])
        
        let initialData = try dataBuilder.getData()
        
        // Deserialize the data
        let dataReader = try Reader(data: initialData)
        print("Initial data:")
        print("Name: \(dataReader.getString("name"))")
        print("Age: \(dataReader.getInt("age"))")
        print("Tags: \(dataReader.getArray("tags"))")
        
        // Create a new map with modified data
        let modifiedBuilder = Builder()
        modifiedBuilder.push("name", "John Doe") // Keep existing field
        // Remove "age" field
        
        // Create updated tags
        modifiedBuilder.pushArray("tags", ["developer", "swift", "flexbuffers"]) // Add new tag
        
        // Add new fields
        modifiedBuilder.push("email", "john@example.com")
        modifiedBuilder.push("active", true)
        
        let modifiedData = try modifiedBuilder.getData()
        
        // Deserialize the modified data
        let modifiedReader = try Reader(data: modifiedData)
        print("\nModified data (added/removed fields):")
        print("Name: \(modifiedReader.getString("name"))")
        
        // Check for removed field
        print("Age present: \(modifiedReader.hasKey("age"))")
        if modifiedReader.hasKey("age") {
            print("Age: \(modifiedReader.getInt("age"))")
        }
        
        // Check for new fields
        print("Email: \(modifiedReader.getString("email"))")
        print("Active: \(modifiedReader.getBool("active"))")
        print("Tags: \(modifiedReader.getArray("tags"))")
        
        // 7. Schema evolution example
        print("\n== Schema evolution example ==")
        
        // Create a product with the original schema
        let productV1 = ProductV1(id: 101, name: "FlexBuff Widget", price: 19.99)
        
        // Serialize using original schema
        let serializedV1 = try FlexBuffers.toData(productV1)
        print("ProductV1 serialized size: \(serializedV1.count) bytes")
        
        // Deserialize using the original schema
        let deserializedV1: ProductV1 = try FlexBuffers.fromData(serializedV1)
        print("ProductV1 deserialized: id=\(deserializedV1.id), name=\(deserializedV1.name), price=\(deserializedV1.price)")
        
        // Deserialize V1 as V2 with custom decoder
        let jsonV1 = try JSONSerialization.jsonObject(with: serializedV1) as! [String: Any]
        let v2FromV1 = ProductV2(
            id: jsonV1["id"] as! UInt32,
            name: jsonV1["name"] as! String,
            price: jsonV1["price"] as! Double
        )
        
        print("\nV1 data read as V2 (forward compatibility):")
        print("id: \(v2FromV1.id)")
        print("name: \(v2FromV1.name)")
        print("price: \(v2FromV1.price)")
        print("description: \(v2FromV1.productDescription ?? "nil")")
        print("in_stock: \(v2FromV1.inStock) (default)")
        print("tags count: \(v2FromV1.tags.count) (default)")
        
        // Create a product with the new schema
        let productV2 = ProductV2(
            id: 102,
            name: "FlexBuff Pro",
            price: 29.99,
            productDescription: "Enhanced FlexBuffer widget with extra features",
            inStock: true,
            tags: ["new", "improved", "featured"]
        )
        
        // Serialize using new schema
        let serializedV2 = try FlexBuffers.toData(productV2)
        print("\nProductV2 serialized size: \(serializedV2.count) bytes")
        
        // Deserialize using the new schema
        let deserializedV2: ProductV2 = try FlexBuffers.fromData(serializedV2)
        print("ProductV2 deserialized:")
        print("id: \(deserializedV2.id)")
        print("name: \(deserializedV2.name)")
        print("price: \(deserializedV2.price)")
        print("description: \(deserializedV2.productDescription ?? "nil")")
        print("inStock: \(deserializedV2.inStock)")
        print("tags: \(deserializedV2.tags)")
        
        // Try to deserialize the V2 data using the V1 schema
        let jsonV2 = try JSONSerialization.jsonObject(with: serializedV2) as! [String: Any]
        let v1FromV2 = ProductV1(
            id: jsonV2["id"] as! UInt32,
            name: jsonV2["name"] as! String,
            price: jsonV2["price"] as! Double
        )
        
        print("\nV2 data read as V1 (backward compatibility):")
        print("id: \(v1FromV2.id)")
        print("name: \(v1FromV2.name)")
        print("price: \(v1FromV2.price)")
        print("(Additional V2 fields are ignored when reading as V1)")
        
    } catch {
        print("Error: \(error)")
    }
}

// Run the main function
main() 