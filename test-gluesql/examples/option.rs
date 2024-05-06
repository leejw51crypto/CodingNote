use anyhow::Result;
use fake::faker::lorem::en::Word;
use fake::Fake;
use gluesql::core::ast_builder::col;
use gluesql::core::ast_builder::table;
use gluesql::core::ast_builder::text;
use gluesql::core::ast_builder::Execute;
use gluesql::prelude::*;

pub struct GlueDatabase {
    glue: Glue<JsonStorage>,
}

impl GlueDatabase {
    pub async fn new(db_path: &str) -> Result<Self> {
        let storage = JsonStorage::new(db_path)?;
        let mut glue = Glue::new(storage);

        glue.execute(
            "CREATE TABLE IF NOT EXISTS options (
                mykey TEXT PRIMARY KEY,
                myvalue TEXT
             )",
        )
        .await?;

        Ok(GlueDatabase { glue })
    }

    pub async fn write_option(&mut self, key: &str, value: &str) -> Result<()> {
        let key2 = key.to_owned();
        let value2 = value.to_owned();

        println!("Debug: Checking if key '{}' exists", key);
        let result = table("options")
            .select()
            .filter(col("mykey").eq(text(key)))
            .project("myvalue")
            .execute(&mut self.glue)
            .await;

        match result {
            Ok(Payload::Select { labels: _, rows }) => {
                if rows.len() > 0 {
                    println!("Debug: Key '{}' exists, performing update", key);
                    table("options")
                        .update()
                        .filter(col("mykey").eq(text(key2)))
                        .set("myvalue", text(value2))
                        .execute(&mut self.glue)
                        .await?;
                } else {
                    println!("Debug: Key '{}' doesn't exist, performing insert", key);
                    table("options")
                        .insert()
                        .columns("mykey, myvalue")
                        .values(vec![vec![text(key2), text(value2)]])
                        .execute(&mut self.glue)
                        .await?;
                }
            }
            _ => {
                println!(
                    "Debug: Select query encountered an error, performing insert for key '{}'",
                    key
                );
                table("options")
                    .insert()
                    .columns("mykey, myvalue")
                    .values(vec![vec![text(key2), text(value2)]])
                    .execute(&mut self.glue)
                    .await?;
            }
        }

        Ok(())
    }

    pub async fn read_option(&mut self, key: &str, defaultvalue: &str) -> Result<String> {
        println!("Debug: Reading value for key '{}'", key);
        let result = table("options")
            .select()
            .filter(col("mykey").eq(text(key)))
            .project("myvalue")
            .execute(&mut self.glue)
            .await?;

        let value = match result {
            Payload::Select { labels: _, rows } => {
                if let Some(row) = rows.into_iter().next() {
                    match &row[0] {
                        Value::Str(s) => s.clone(),
                        _ => anyhow::bail!("Unsupported value type"),
                    }
                } else {
                    println!("Debug: Key '{}' not found, using default value", key);
                    defaultvalue.to_owned()
                }
            }
            _ => anyhow::bail!("Unexpected payload type"),
        };

        Ok(value)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut db = GlueDatabase::new("mydatabase.json").await?;

    let fake_word1: String = Word().fake();
    let fake_word2: String = Word().fake();

    db.write_option("key1", &fake_word1).await?;
    db.write_option("key2", &fake_word2).await?;

    let value1 = db.read_option("key1", "default").await?;
    println!("Value for key1: {:?}", value1);
    assert_eq!(
        value1, fake_word1,
        "Read value does not match written value for key1"
    );

    let value2 = db.read_option("key2", "default").await?;
    println!("Value for key2: {:?}", value2);
    assert_eq!(
        value2, fake_word2,
        "Read value does not match written value for key2"
    );

    Ok(())
}
