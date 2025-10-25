@0xdbb9ad1f14bf0b36;  # unique file ID

interface Calculator {
    # Simple arithmetic operations
    add @0 (a :Int32, b :Int32) -> (result :Int32);
    subtract @1 (a :Int32, b :Int32) -> (result :Int32);
    multiply @2 (a :Int32, b :Int32) -> (result :Int32);
    divide @3 (a :Int32, b :Int32) -> (result :Float32);

    # Advanced operation with async callback
    evaluate @4 (expression :Text) -> (result :Float32);
}
