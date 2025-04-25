@0xf8f358c75ba16287;  # Unique 64-bit ID for the schema

struct HelloWorldData {
    message @0 :Text;
    timestamp @1 :Text;
    doubleValue @2 :Float64;  # float64
    intArray @3 :List(Int64);
    binaryData @4 :Data; 
}

interface Hello {
  # Get current time in both local and UTC
  getCurrentTime @0 () -> (localTime :Text, utcTime :Text);

  # Simple greeting with a name
  greeting @1 (name :Text) -> (message :Text);

  # Just return "Hello, World!" with current time
  helloWorld @2 (count :Int64) -> (result :HelloWorldData);
}
