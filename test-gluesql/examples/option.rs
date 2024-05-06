use anyhow::anyhow;
use anyhow::Result;
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
        // Check if the key exists
        let query = format!("SELECT myvalue FROM options WHERE mykey = '{}'", key);
        let result = self.glue.execute(&query).await?;

        let payload = result
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to read option"))?;

        if let Some(_) = payload
            .select()
            .ok_or(anyhow::anyhow!("Cannot select"))?
            .next()
        {
            // Key exists, perform an update
            let update_query = format!(
                "UPDATE options SET myvalue = '{}' WHERE mykey = '{}'",
                value, key
            );
            self.glue.execute(&update_query).await?;
        } else {
            // Key doesn't exist, perform an insert
            let insert_query = format!(
                "INSERT INTO options (mykey, myvalue) VALUES ('{}', '{}')",
                key, value
            );
            self.glue.execute(&insert_query).await?;
        }

        Ok(())
    }

    pub async fn read_option(&mut self, key: &str, defaultvalue: &str) -> Result<String> {
        let query = format!("SELECT myvalue FROM options WHERE mykey = '{}'", key);
        let result = self.glue.execute(&query).await?;

        let payload = result
            .into_iter()
            .next()
            .ok_or(anyhow!("cannot get data"))?;
        let value = match payload
            .select()
            .ok_or(anyhow::anyhow!("Failed to select payload"))?
            .next()
        {
            Some(map) => {
                let value = map
                    .get("myvalue")
                    .ok_or(anyhow::anyhow!("Failed to get 'myvalue' from map"))?;
                match value {
                    Value::Str(s) => s.clone(),
                    _ => anyhow::bail!("Unsupported value type"),
                }
            }
            None => defaultvalue.to_owned(),
        };

        Ok(value)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut db = GlueDatabase::new("mydatabase.json").await?;

    db.write_option("key1", "value1").await?;
    db.write_option("key2", "value2").await?;

    let value1 = db.read_option("key1", "0").await?;
    println!("Value for key1: {:?}", value1);

    let value2 = db.read_option("key2", "0").await?;
    println!("Value for key2: {:?}", value2);

    Ok(())
}
