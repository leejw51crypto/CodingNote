@0xdff9154bc1cf6733;  # Unique file ID

struct Author {
  name @0 :Text;
  email @1 :Text;
}

struct Book {
  title @0 :Text;
  author @1 :Author;
}

struct Library {
  name @0 :Text;
  books @1 :List(Book);
}

