# hello.capnp
@0xfb9d8ad1bd916d7f;

struct HelloWorld {
  message @0 :Text;
  video @1: Data;
}

struct Address {
    street @0 :Text;
    city @1 :Text;
    zip @2 :Text;
}

struct Person {
    name @0 :Text;
    age @1 : Int32;
    address @2 :Address;
    photo @3: Data;
}

struct People {
    persons @0 :List(Person);
}