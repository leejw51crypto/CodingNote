@0xd1d3f48382ab2e4d;  # Unique file ID
struct Car {
  make @0 :Text;
  model @1 :Text;
  year @2 :UInt16;
  color @3 :Text;
  mileage @4 :UInt32;
  vin @5 :Text;
  isElectric @6 :Bool;
  price @7 :UInt32;
}


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

struct Shop {
  name @0 :Text;
  address @1 :Text;
  books @2 :List(Book);
  fruits @3 :List(Fruit);
  isOpen @4 :Bool;
  openingHours @5 :Text;
  phoneNumber @6 :Text;
  email @7 :Text;
  rating @8 :Float32;
  lastUpdated @9 :Int64;  # Unix timestamp
}
