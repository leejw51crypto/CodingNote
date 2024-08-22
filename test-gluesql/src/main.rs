use gluesql::prelude::Value;
use {
    chrono::Utc,
    fake::faker::address::en::SecondaryAddress,
    fake::{faker::name::en::Name, faker::phone_number::en::PhoneNumber, Fake},
    gluesql::{prelude::Glue, },
};
use gluesql::gluesql_sled_storage::SledStorage;

struct GreetRow {
    id: i64,
    name: String,
    address: String,
    phone_number: String,
    timestamp: String,
}

pub async fn run() {
    // Initiate a connection
    // Open a Sled database, this will create one if one does not yet exist
    let sled_dir = "/tmp/gluesql/hello_world";
    let storage = SledStorage::new(sled_dir).expect("Something went wrong!");
    // Wrap the Sled database with Glue
    let mut glue = Glue::new(storage);

    // Create table if it doesn't exist
    let queries = "
            CREATE TABLE IF NOT EXISTS greet (
                id INTEGER PRIMARY KEY,
                name TEXT,
                address TEXT,
                phone_number TEXT,
                timestamp TEXT
            );
        ";

    glue.execute(queries).await.expect("Execution failed");

    // Insert new rows with fake data and current timestamp
    let fake_data = (0..1)
        .map(|i| {
            let name: String = Name().fake();
            let address: String = SecondaryAddress().fake();
            let phone_number: String = PhoneNumber().fake();
            let current_timestamp = Utc::now().to_string();
            // get id  from chrono timestamp
            let myid :i64= Utc::now().timestamp();


            format!(
                "INSERT INTO greet (id, name, address, phone_number, timestamp) VALUES ({}, '{}', '{}', '{}', '{}');",
                myid, name, address, phone_number, current_timestamp
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    glue.execute(&fake_data).await.expect("Execution failed");

    // Select all rows
    let queries = "
            SELECT id, name, address, phone_number, timestamp FROM greet
        ";

    let mut result = glue.execute(queries).await.expect("Failed to execute");

    assert_eq!(result.len(), 1);

    let payload = result.remove(0);

    let rows = payload
        .select()
        .unwrap()
        .map(|map| {
            let id = *map.get("id").unwrap();
            let id = match *map.get("id").unwrap() {
                Value::I64(v) => v,

                _ => panic!("Unsupported id type"),
            };
            let name = *map.get("name").unwrap();
            let name = name.into();
            let address = *map.get("address").unwrap();
            let address = address.into();
            let phone_number = *map.get("phone_number").unwrap();
            let phone_number = phone_number.into();
            let timestamp = *map.get("timestamp").unwrap();
            let timestamp = timestamp.into();

            GreetRow {
                id: *id,
                name,
                address,
                phone_number,
                timestamp,
            }
        })
        .collect::<Vec<_>>();

    println!("Greetings:");
    for row in &rows {
        println!(
            "Hello, {}! (ID: {}, {}, {}, {})",
            row.name, row.id, row.address, row.phone_number, row.timestamp
        );
    }
}

#[tokio::main]
async fn main() {
    run().await;
}
