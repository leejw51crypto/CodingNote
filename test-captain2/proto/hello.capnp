@0x9663f4dd604afa35;

interface HelloWorld {
    struct MyData {
        name @0 :Text;
        age @1 :Int64;
        disk @2 : Data;
        mynotes @3 : List(Text);
    }
    struct HelloRequest {
        name @0 :Text;
    }

    struct HelloReply {
        message @0 :Text;
    }

    sayHello @0 (request: HelloRequest) -> (reply: HelloReply);
}