use fake::faker::name::en::Name;
use fake::Fake;
use rusqlite::{Connection, Result};

fn main() -> Result<()> {
    // Connect to a SQLite database file (it will be created if it doesn't exist)
    let conn = Connection::open("usersbasic.db")?;

    // Create a table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)",
        [],
    )?;

    // Insert 10 rows with fake names into the table
    for _ in 0..10 {
        let fake_name: String = Name().fake();
        conn.execute("INSERT INTO users (name) VALUES (?)", [fake_name])?;
    }

    // Query the table
    let mut stmt = conn.prepare("SELECT * FROM users")?;
    let user_iter = stmt.query_map([], |row| {
        Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?))
    })?;

    // Print the results
    for user in user_iter {
        let (id, name) = user.unwrap();
        println!("ID: {}, Name: {}", id, name);
    }

    Ok(())
}
