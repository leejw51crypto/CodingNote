use fake::faker::internet::en::FreeEmail;
use fake::faker::name::en::{FirstName, LastName};
use fake::{Fake, Faker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: u32,
    name: String,
    email: String,
    active: bool,
    scores: Vec<i32>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Address {
    street: String,
    city: String,
    country: String,
    postal_code: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UserWithAddress {
    user: User,
    address: Address,
    created_at: u64,
}

impl User {
    fn generate_fake() -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("registration_date".to_string(), "2023-05-15".to_string());
        metadata.insert("last_login".to_string(), "2023-09-20".to_string());

        User {
            id: (1..10000).fake(),
            name: format!(
                "{} {}",
                FirstName().fake::<String>(),
                LastName().fake::<String>()
            ),
            email: FreeEmail().fake(),
            active: rand::random::<bool>(),
            scores: vec![(1..100).fake(), (1..100).fake(), (1..100).fake()],
            metadata,
        }
    }
}

impl Address {
    fn generate_fake() -> Self {
        Address {
            street: format!("{} Main St", (1..1000).fake::<u16>()),
            city: ["New York", "London", "Tokyo", "Paris", "Berlin"]
                .choose()
                .to_string(),
            country: ["USA", "UK", "Japan", "France", "Germany"]
                .choose()
                .to_string(),
            postal_code: format!("{:05}", (10000..99999).fake::<u32>()),
        }
    }
}

impl UserWithAddress {
    fn generate_fake() -> Self {
        UserWithAddress {
            user: User::generate_fake(),
            address: Address::generate_fake(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

fn main() {
    // 1. Basic example: Generate fake user data
    let user = User::generate_fake();
    println!("Original user data: {:#?}", user);

    // Serialize to FlexBuffer
    let serialized = flexbuffers::to_vec(&user).expect("Failed to serialize");
    println!("Serialized data size: {} bytes", serialized.len());

    // Deserialize from FlexBuffer
    let deserialized: User = flexbuffers::from_slice(&serialized).expect("Failed to deserialize");
    println!("Deserialized user data: {:#?}", deserialized);

    // 2. Nested structures example
    let users = vec![
        User::generate_fake(),
        User::generate_fake(),
        User::generate_fake(),
    ];

    // Serialize the vector of users
    let serialized_users = flexbuffers::to_vec(&users).expect("Failed to serialize users");
    println!(
        "Serialized users data size: {} bytes",
        serialized_users.len()
    );

    // Deserialize the vector of users
    let deserialized_users: Vec<User> =
        flexbuffers::from_slice(&serialized_users).expect("Failed to deserialize users");
    println!("Deserialized {} users", deserialized_users.len());

    // 3. Manual serialization using the FlexBufferBuilder
    let mut builder = flexbuffers::Builder::default();
    {
        let mut map = builder.start_map();
        map.push("int_value", 42);
        map.push("float_value", 3.14159);
        map.push("string_value", "hello flexbuffers");
        map.push("bool_value", true);

        let mut vec = map.start_vector("array_value");
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.end_vector();

        map.end_map();
    }
    let manual_serialized = builder.view();

    // Manual deserialization using the Reader
    let reader = flexbuffers::Reader::get_root(manual_serialized).expect("Failed to get root");
    println!("Manual deserialization results:");

    // Access map values
    let map = reader.as_map();
    println!("int_value: {}", map.idx("int_value").as_i32());
    println!("float_value: {}", map.idx("float_value").as_f64());
    println!("string_value: {}", map.idx("string_value").as_str());
    println!("bool_value: {}", map.idx("bool_value").as_bool());

    // Access vector
    let array = map.idx("array_value").as_vector();
    print!("array_value: [");
    for i in 0..array.len() {
        print!("{}", array.idx(i).as_i32());
        if i < array.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // 4. Complex nested structure with custom types
    let user_with_address = UserWithAddress::generate_fake();
    println!("\nComplex user with address: {:#?}", user_with_address);

    // Serialize to FlexBuffer
    let serialized_complex =
        flexbuffers::to_vec(&user_with_address).expect("Failed to serialize complex");
    println!(
        "Serialized complex data size: {} bytes",
        serialized_complex.len()
    );

    // Deserialize from FlexBuffer
    let deserialized_complex: UserWithAddress =
        flexbuffers::from_slice(&serialized_complex).expect("Failed to deserialize complex");
    println!(
        "Successfully deserialized complex structure with timestamp: {}",
        deserialized_complex.created_at
    );

    // 5. Binary data example
    let binary_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

    // Serialize binary data directly as a vector
    let serialized_binary = flexbuffers::to_vec(&binary_data).expect("Failed to serialize binary");

    // Deserialize binary data
    let binary_vec: Vec<u8> =
        flexbuffers::from_slice(&serialized_binary).expect("Failed to deserialize binary");

    print!("\nOriginal binary: [");
    for (i, byte) in binary_data.iter().enumerate() {
        print!("0x{:02X}", byte);
        if i < binary_data.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    print!("Deserialized binary: [");
    for (i, byte) in binary_vec.iter().enumerate() {
        print!("0x{:02X}", byte);
        if i < binary_vec.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // 6. Data manipulation example - adding and removing fields
    println!("\n== Data manipulation example ==");

    // Create a map with initial data
    let mut data_builder = flexbuffers::Builder::default();
    {
        let mut map = data_builder.start_map();
        map.push("name", "John Doe");
        map.push("age", 30);

        // Create a vector for tags
        let mut tags_vec = map.start_vector("tags");
        tags_vec.push("developer");
        tags_vec.push("rust");
        tags_vec.end_vector();

        map.end_map();
    }
    let initial_data = data_builder.view();

    // Deserialize the data
    let reader = flexbuffers::Reader::get_root(initial_data).expect("Failed to read initial data");
    let map = reader.as_map();

    println!("Initial data:");
    println!("Name: {}", map.idx("name").as_str());
    println!("Age: {}", map.idx("age").as_i32());

    let tags = map.idx("tags").as_vector();
    print!("Tags: [");
    for i in 0..tags.len() {
        print!("{}", tags.idx(i).as_str());
        if i < tags.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Create a new map with modified data (backward compatible)
    // Adding new fields and removing old ones
    let mut modified_builder = flexbuffers::Builder::default();
    {
        let mut map = modified_builder.start_map();
        map.push("name", "John Doe"); // Keep existing field
        // Remove "age" field

        // Create a vector for tags with an additional tag
        let mut tags_vec = map.start_vector("tags");
        tags_vec.push("developer");
        tags_vec.push("rust");
        tags_vec.push("flexbuffers"); // Add a new tag
        tags_vec.end_vector();

        map.push("email", "john@example.com"); // Add new field
        map.push("active", true); // Add new field
        map.end_map();
    }
    let modified_data = modified_builder.view();

    // Deserialize the modified data
    let reader =
        flexbuffers::Reader::get_root(modified_data).expect("Failed to read modified data");
    let map = reader.as_map();

    println!("\nModified data (added/removed fields):");
    println!("Name: {}", map.idx("name").as_str());

    // Check for removed field
    println!(
        "Age present: {}",
        map.idx("age").flexbuffer_type() != flexbuffers::FlexBufferType::Null
    );
    if map.idx("age").flexbuffer_type() != flexbuffers::FlexBufferType::Null {
        println!("Age: {}", map.idx("age").as_i32());
    }

    // Check for new fields
    println!("Email: {}", map.idx("email").as_str());
    println!("Active: {}", map.idx("active").as_bool());

    let tags = map.idx("tags").as_vector();
    print!("Tags: [");
    for i in 0..tags.len() {
        print!("{}", tags.idx(i).as_str());
        if i < tags.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // 7. Schema evolution example - adding fields while maintaining compatibility
    println!("\n== Schema evolution example ==");

    // Original schema: Simple product
    #[derive(Debug, Serialize, Deserialize)]
    struct ProductV1 {
        id: u32,
        name: String,
        price: f64,
    }

    // Evolved schema: Product with additional fields
    #[derive(Debug, Serialize, Deserialize)]
    struct ProductV2 {
        id: u32,
        name: String,
        price: f64,
        // New fields with default values
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        in_stock: bool,
        #[serde(default)]
        tags: Vec<String>,
    }

    // Create a product with the original schema
    let product_v1 = ProductV1 {
        id: 101,
        name: "FlexBuff Widget".to_string(),
        price: 19.99,
    };

    // Serialize using original schema
    let serialized_v1 = flexbuffers::to_vec(&product_v1).expect("Failed to serialize ProductV1");
    println!("ProductV1 serialized size: {} bytes", serialized_v1.len());

    // Deserialize using the original schema
    let deserialized_v1: ProductV1 =
        flexbuffers::from_slice(&serialized_v1).expect("Failed to deserialize ProductV1");
    println!(
        "ProductV1 deserialized: id={}, name={}, price={}",
        deserialized_v1.id, deserialized_v1.name, deserialized_v1.price
    );

    // Now try to deserialize the V1 data using the V2 schema - demonstrating forward compatibility
    let deserialized_as_v2: ProductV2 =
        flexbuffers::from_slice(&serialized_v1).expect("Failed to deserialize as ProductV2");
    println!("\nV1 data read as V2 (forward compatibility):");
    println!("id: {}", deserialized_as_v2.id);
    println!("name: {}", deserialized_as_v2.name);
    println!("price: {}", deserialized_as_v2.price);
    println!("description: {:?}", deserialized_as_v2.description);
    println!("in_stock: {} (default)", deserialized_as_v2.in_stock);
    println!("tags count: {} (default)", deserialized_as_v2.tags.len());

    // Create a product with the new schema
    let product_v2 = ProductV2 {
        id: 102,
        name: "FlexBuff Pro".to_string(),
        price: 29.99,
        description: Some("Enhanced FlexBuffer widget with extra features".to_string()),
        in_stock: true,
        tags: vec![
            "new".to_string(),
            "improved".to_string(),
            "featured".to_string(),
        ],
    };

    // Serialize using new schema
    let serialized_v2 = flexbuffers::to_vec(&product_v2).expect("Failed to serialize ProductV2");
    println!("\nProductV2 serialized size: {} bytes", serialized_v2.len());

    // Deserialize using the new schema
    let deserialized_v2: ProductV2 =
        flexbuffers::from_slice(&serialized_v2).expect("Failed to deserialize ProductV2");
    println!("ProductV2 deserialized:");
    println!("id: {}", deserialized_v2.id);
    println!("name: {}", deserialized_v2.name);
    println!("price: {}", deserialized_v2.price);
    println!("description: {:?}", deserialized_v2.description);
    println!("in_stock: {}", deserialized_v2.in_stock);
    print!("tags: [");
    for (i, tag) in deserialized_v2.tags.iter().enumerate() {
        print!("{}", tag);
        if i < deserialized_v2.tags.len() - 1 {
            print!(", ");
        }
    }
    println!("]");

    // Now try to deserialize the V2 data using the V1 schema - demonstrating backward compatibility
    let deserialized_back_to_v1: ProductV1 =
        flexbuffers::from_slice(&serialized_v2).expect("Failed to deserialize back to ProductV1");
    println!("\nV2 data read as V1 (backward compatibility):");
    println!("id: {}", deserialized_back_to_v1.id);
    println!("name: {}", deserialized_back_to_v1.name);
    println!("price: {}", deserialized_back_to_v1.price);
    println!("(Additional V2 fields are ignored when reading as V1)");
}

trait Choose<T> {
    fn choose(&self) -> T;
}

impl<T: Clone> Choose<T> for [T] {
    fn choose(&self) -> T {
        let idx = (rand::random::<f32>() * self.len() as f32) as usize;
        self[idx % self.len()].clone()
    }
}
