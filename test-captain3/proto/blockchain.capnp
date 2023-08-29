# blockchain.capnp
@0xfab8475b5ed6be35;

struct Transaction {
  sender @0 :Text;
  recipient @1 :Text;
  amount @2 :Int64;
}

struct Block {
  timestamp @0 :Int64;
  transactions @1 :List(Transaction);
  prevHash @2 :Data;
}

struct Block2 {
  block @0 :Data;
  hash @1 :Data;
}

struct Blockchain {
  blocks @0 :List(Block2);
}
