use fake::faker::internet::en::SafeEmail;
use fake::faker::name::en::Name;
use fake::faker::phone_number::en::PhoneNumber;
use fake::Fake;
use rusqlite::{Connection, Result};

fn main() -> Result<()> {
    // Connect to a SQLite database file (it will be created if it doesn't exist)
    let conn = Connection::open("users.db")?;

    // Create a table with additional fields
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT
        )",
        [],
    )?;

    // Insert 10 rows with fake data into the table
    for _ in 0..10 {
        let fake_name: String = Name().fake();
        let fake_email: String = SafeEmail().fake();
        let fake_phone: String = PhoneNumber().fake();

        conn.execute(
            "INSERT INTO users (name, email, phone) VALUES (?, ?, ?)",
            [fake_name, fake_email, fake_phone],
        )?;
    }

    // Query the table
    let mut stmt = conn.prepare("SELECT * FROM users")?;
    let user_iter = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i32>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;

    // Print the results
    for user in user_iter {
        let (id, name, email, phone) = user.unwrap();
        println!(
            "ID: {}, Name: {}, Email: {}, Phone: {}",
            id, name, email, phone
        );
    }

    Ok(())
}
