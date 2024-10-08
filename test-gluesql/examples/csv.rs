use anyhow::Result;
use base64::engine::general_purpose;
use base64::Engine;
use chrono::Utc;
use gluesql::core::ast_builder::{col, num, table, text, Execute};
use gluesql::prelude::*;
// use CsvStorage;
use gluesql_csv_storage::CsvStorage;
#[tokio::main]
async fn main() -> Result<()> {
    let storage = CsvStorage::new("mydatabase.csv")?;

    let mut glue = Glue::new(storage);

    let create_table = r#"
    CREATE TABLE IF NOT EXISTS my_table (
        id INTEGER PRIMARY KEY,
        data TEXT
    )"#;
    glue.execute(create_table).await?;

    let byte_data = "apple hello world".as_bytes();
    let encoded_data = general_purpose::STANDARD.encode(byte_data);

    let timestamp = Utc::now().timestamp_millis();

    table("my_table")
        .insert()
        .columns("id,data")
        .values(vec![vec![num(timestamp as i64), text(encoded_data)]])
        .execute(&mut glue)
        .await?;

    let query = table("my_table")
        .select()
        .project(col("id"))
        .project(col("data"));

    let result = query.execute(&mut glue).await?;

    if let Payload::Select { labels: _, rows } = result {
        for row in rows {
            if let (Value::I64(id), Value::Str(data)) = (&row[0], &row[1]) {
                let decoded_data = general_purpose::STANDARD.decode(data)?;
                let restored_string = String::from_utf8(decoded_data)?;
                println!("ID: {}, Restored string: {}", id, restored_string);
            }
        }
    }

    Ok(())
}
