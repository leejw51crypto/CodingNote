use anyhow::Result;
use chrono::Utc;
use gluesql::core::ast_builder::{col, num, table, text, Execute};
use gluesql::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let storage = JsonStorage::new("mydatabase.json")?;
    let mut glue = Glue::new(storage);

    let create_table = r#"
        CREATE TABLE IF NOT EXISTS my_table (
            id INTEGER PRIMARY KEY,
            data BYTEA
        )"#;
    glue.execute(create_table).await?;

    let byte_data = "apple hello world";
    let timestamp = Utc::now().timestamp_millis();

    table("my_table")
        .insert()
        .columns("id,data")
        .values(vec![vec![
            num(timestamp as i64),
            text(hex::encode(&byte_data)),
        ]])
        .execute(&mut glue)
        .await?;

    let query = table("my_table")
        .select()
        .project(col("id"))
        .project(col("data"));
    let result = query.execute(&mut glue).await?;
    if let Payload::Select { labels: _, rows } = result {
        for row in rows {
            if let (Value::I64(id), Value::Bytea(data)) = (&row[0], &row[1]) {
                let restored_string = String::from_utf8(data.clone())?;
                println!("ID: {}, Restored string: {}", id, restored_string);
            }
        }
    }

    Ok(())
}
