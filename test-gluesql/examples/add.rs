use anyhow::Result;
use gluesql::prelude::Value;
use {
    chrono::Utc,
    fake::faker::address::en::SecondaryAddress,
    fake::{faker::name::en::Name, faker::phone_number::en::PhoneNumber, Fake},
    gluesql::{prelude::Glue, sled_storage::SledStorage},
};
struct GreetRow {
    name: String,
    address: String,
    phone_number: String,
    timestamp: String,
}

async fn write_option(glue: &mut Glue<SledStorage>, key: &str, value: &str) -> Result<()> {
    println!("write option {} {}", key, value);

    // Check if the key exists
    let query = format!("SELECT myvalue FROM options WHERE mykey = '{}'", key);
    let result = glue.execute(&query).await?;

    if let Some(payload) = result.into_iter().next() {
        // print payload
        println!("payload {:?}", payload);
        match payload
            .select()
            .ok_or(anyhow::anyhow!("Failed to select payload"))?
            .next()
        {
            Some(existing_value) => {
                println!("existing value {:?}", existing_value);
                // Key exists, perform an update
                let update_query = format!(
                    "UPDATE options SET myvalue = '{}' WHERE mykey = '{}'",
                    value, key
                );
                println!("update query {}", update_query);
                glue.execute(&update_query).await?;
            }
            None => {
                // Key doesn't exist, perform an insert
                let insert_query = format!(
                    "INSERT INTO options (mykey, myvalue) VALUES ('{}', '{}')",
                    key, value
                );
                println!("insert query {}", insert_query);
                glue.execute(&insert_query).await?;
            }
        }
    } else {
        anyhow::bail!("Failed to get payload");
    }

    Ok(())
}

async fn read_option(
    glue: &mut Glue<SledStorage>,
    key: &str,
    defaultvalue: &str,
) -> Result<String> {
    println!("read option {} default {}", key, defaultvalue);
    let query = format!("SELECT myvalue FROM options WHERE mykey = '{}'", key);
    let result = glue.execute(&query).await?;

    let value = if let Some(payload) = result.into_iter().next() {
        match payload
            .select()
            .ok_or(anyhow::anyhow!("Failed to select payload"))?
            .next()
        {
            Some(map) => {
                println!("map {:?}", map.get("myvalue"));
                let value = map
                    .get("myvalue")
                    .ok_or(anyhow::anyhow!("Failed to get 'myvalue' from map"))?;
                match value {
                    Value::Null => "null".to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::I64(i) => i.to_string(),
                    Value::F64(f) => f.to_string(),
                    Value::Str(s) => s.clone(),
                    Value::Bytea(b) => String::from_utf8_lossy(b).to_string(),
                    Value::Date(d) => d.to_string(),
                    Value::Time(t) => t.to_string(),
                    _ => anyhow::bail!("Unsupported value type"),
                }
            }
            None => defaultvalue.to_owned(),
        }
    } else {
        defaultvalue.to_owned()
    };

    println!("read option {} value {} ok", key, value);
    Ok(value)
}

pub async fn run() {
    // Initiate a connection
    // Open a Sled database, this will create one if one does not yet exist
    let sled_dir = "/tmp/gluesql/hello_world3";
    let storage = SledStorage::new(sled_dir).expect("Something went wrong!");
    // Wrap the Sled database with Glue
    let mut glue = Glue::new(storage);

    // Create table if it doesn't exist
    let queries = "
            CREATE TABLE IF NOT EXISTS greet (
                name TEXT,
                address TEXT,
                phone_number TEXT,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS options (
                mykey TEXT,
                myvalue TEXT
            );
        ";

    glue.execute(queries).await.expect("Execution failed");

    // Test write_option and read_option
    write_option(&mut glue, "key1", "value1")
        .await
        .expect("Failed to write option");
    let value = read_option(&mut glue, "key1", "default")
        .await
        .expect("Failed to read option");
    assert_eq!(value, "value1");
}

#[tokio::main]
async fn main() {
    run().await;
}
