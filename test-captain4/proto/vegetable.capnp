@0x8e739edd178d9eb1;  # Unique file ID

struct Vegetable {
  struct Details {
    struct Nutrition {
      calories @0 :UInt16;
      protein @1 :Float32;
      fiber @2 :Float32;
      vitamins @3 :List(Vitamin);
      minerals @4 :List(Text);
    }

    struct Vitamin {
      name @0 :Text;
      amount @1 :Float32;
      unit @2 :Text;
    }

    name @0 :Text;
    scientificName @1 :Text;
    color @2 :Text;
    shape @3 :Text;
    nutrition @4 :Nutrition;
    growthTime @5 :UInt16;  # in days
    preferredSoilPH @6 :Float32;
    harvestMethod @7 :Text;
  }

  details @0 :Details;
  id @1 :UInt32;
  inSeason @2 :Bool;
  plantDate @3 :Date;
  expectedHarvestDate @4 :Date;
  price @5 :Float64;
  quantity @6 :UInt32;
  organic @7 :Bool;
  supplier @8 :Text;
  tags @9 :List(Text);
  lastWatered @10 :Time;
  image @11 :Data;
}

struct Date {
  year @0 :Int16;
  month @1 :UInt8;
  day @2 :UInt8;
}

struct Time {
  hour @0 :UInt8;
  minute @1 :UInt8;
  second @2 :UInt8;
}