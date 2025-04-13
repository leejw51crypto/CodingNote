@0xd1d3f48382ab2e4d;  # Unique file ID

struct Book {
  title @0 :Text;
  author @1 :Text;
  pages @2 :UInt32;
  publishYear @3 :UInt16;
  genres @4 :List(Text);
  isAvailable @5 :Bool;
}

struct Fruit {
  name @0 :Text;
  color @1 :Text;
  weightGrams @2 :UInt32;
  isRipe @3 :Bool;
  variety @4 :Text;
}

struct Message {
  union {
    book @0 :Book;
    fruit @1 :Fruit;
  }
}

struct BookList {
  books @0 :List(Book);
}
