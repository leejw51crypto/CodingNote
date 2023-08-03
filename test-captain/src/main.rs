extern crate capnp;

mod addressbook_capnp {
    include!(concat!(env!("OUT_DIR"), "/src/addressbook_capnp.rs"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut message = capnp::message::Builder::new_default();
    let mut address_book = message.init_root::<addressbook_capnp::address_book::Builder>();

    let mut people = address_book.init_people(1);
    let mut person = people.get(0);
    person.set_name("Alice");
    person.set_email("alice@example.com");

    let mut output_file = std::fs::File::create("addressbook.bin")?;
    capnp::serialize_packed::write_message(&mut output_file, &message)?;

    Ok(())
}
