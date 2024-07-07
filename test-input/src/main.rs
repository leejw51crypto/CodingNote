use anyhow::Result;
use rustyline::DefaultEditor;
fn main() -> Result<()> {
    let mut rl = DefaultEditor::new()?;
    let readline = rl.readline(">> ")?.trim().to_string();
    println!("Line: {}", readline);
    let readline2 = rl.readline(">> ")?.trim().to_string();
    println!("Line2: {}", readline2);
    Ok(())
}
