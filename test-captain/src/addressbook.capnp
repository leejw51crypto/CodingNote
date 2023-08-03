@0x9b7f0be16be66f8b;

struct Person {
  name @0 :Text;
  email @1 :Text;
}

struct AddressBook {
  people @0 :List(Person);
}
