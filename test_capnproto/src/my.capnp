@0x91c48bdfec35ea8c;

struct MultimediaFile {
  filename @0 :Text;
  content @1 :Data;
  size @2 :Int64;
  timestamp @3 :Int64;
  metadata @4 :Metadata;
  tags @5 :List(Text);
  isPublic @6 :Bool;
}

struct Metadata {
  author @0 :Text;
  description @1 :Text;
  creationDate @2 :Int64;
  location @3 :Text;
}
