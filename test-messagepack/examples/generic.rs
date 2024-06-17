use serde::{Serialize, Deserialize};
use rmp_serde::{Serializer};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
enum Node {
    Value(i32),
    Children(Vec<Node>),
}

fn main() {
    let data = Node::Children(vec![
        Node::Value(1),
        Node::Value(2),
        Node::Children(vec![
            Node::Value(3),
            Node::Value(4),
        ]),
    ]);

    // Serialize to JSON
    let json_data = serde_json::to_string(&data).unwrap();
    let json_length = json_data.len();
    println!("JSON serialized data: {}", json_data);
    println!("JSON encoded length: {} bytes", json_length);

    // Serialize to MessagePack
    let mut msgpack_buf = Vec::new();
    data.serialize(&mut Serializer::new(&mut msgpack_buf)).unwrap();
    let msgpack_length = msgpack_buf.len();
    println!("MessagePack serialized data: {:?}", msgpack_buf);
    println!("MessagePack encoded length: {} bytes", msgpack_length);

    // Deserialize MessagePack data without schema
    let msgpack_value: Value = rmp_serde::from_read(&mut &msgpack_buf[..]).unwrap();
    println!("MessagePack deserialized data: {:?}", msgpack_value);

    // Access values from the deserialized MessagePack data
    if let Value::Array(array) = msgpack_value {
        for value in array {
            match value {
                Value::Number(int) => {
                    println!("Integer value: {}", int);
                }
                Value::Array(children) => {
                    println!("Children:");
                    for child in children {
                        if let Value::Number(child_int) = child {
                            println!("  - Integer value: {}", child_int);
                        }
                    }
                }
                _ => {
                    println!("Unexpected value: {:?}", value);
                }
            }
        }
    }
}