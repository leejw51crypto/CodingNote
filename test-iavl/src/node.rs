use anyhow::bail;
use anyhow::Result;
use bytes::BufMut;
use bytes::{Bytes, BytesMut};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use sled::Db;
use std::cmp::Ordering;
use text_trees::StringTreeNode;
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Node {
    pub key: i64,
    pub value: String,

    pub leftnode: Bytes,  // 32 bytes hash
    pub rightnode: Bytes, // 32 bytes hash
}

// implement Display for Node
impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let myhash = self.get_hash().unwrap();
        let left = if self.leftnode.is_empty() {
            Bytes::from(vec![0; 32])
        } else {
            self.leftnode.clone()
        };
        let right = if self.rightnode.is_empty() {
            Bytes::from(vec![0; 32])
        } else {
            self.rightnode.clone()
        };
        write!(
            f,
            "hash:{} left:{} right:{} key:{} value:{}",
            hex::encode(&myhash[0..4]),
            hex::encode(&left[0..4]),
            hex::encode(&right[0..4]),
            self.key,
            self.value,
        )
    }
}

impl Node {
    pub fn new_number_node(db: &Db, key: i64) -> Result<Node> {
        // convert key to string
        let value = format!("value_{}", key);
        let ret = Node {
            key,
            value,
            leftnode: Bytes::new(),
            rightnode: Bytes::new(),
        };

        ret.write(db)?;

        Ok(ret)
    }

    pub fn recursive_get(&mut self, db: &Db, key: i64) -> Result<String> {
        if self.key == key {
            return Ok(self.value.clone());
        }
        if key < self.key {
            if self.leftnode.is_empty() {
                bail!("key not found");
            } else {
                let mut leftnode = self.get_leftnode(db)?;
                return leftnode.recursive_get(db, key);
            }
        }
        if key > self.key {
            if self.rightnode.is_empty() {
                bail!("key not found");
            } else {
                let mut rightnode = self.get_rightnode(db)?;
                return rightnode.recursive_get(db, key);
            }
        }
        bail!("key not found")
    }
    pub fn recursive_set(&mut self, db: &Db, key: i64, value: String) -> Result<Bytes> {
        println!(
            "recursive set node {} key {}  value {}",
            self.key, key, value
        );
        let mut finalhash;

        if self.leftnode.is_empty() && self.rightnode.is_empty() {
            let newnode = Node::new_node(db, key, value.clone())?;
            if key < self.key {
                self.link_left(db, &newnode)?;
                finalhash = self.get_hash()?;
                return Ok(finalhash);
            }
            if key > self.key {
                self.link_right(db, &newnode)?;
                finalhash = self.get_hash()?;
                return Ok(finalhash);
            } else {
                self.value = value;
                finalhash = self.get_hash()?;
                return Ok(finalhash);
            }
        }

        match key.cmp(&self.key) {
            Ordering::Equal => {
                // replace
                self.value = value;
            }
            Ordering::Less => {
                //key < self.key
                if !self.leftnode.is_empty() {
                    let mut leftnode = self.get_leftnode(db)?;
                    let leftnodehash = leftnode.recursive_set(db, key, value)?;
                    self.leftnode = leftnodehash;
                } else {
                    let newnode = Node::new_node(db, key, value)?;
                    self.link_left(db, &newnode)?;
                }
            }
            Ordering::Greater => {
                // key > self.key
                if !self.rightnode.is_empty() {
                    let mut rightnode = self.get_rightnode(db)?;
                    let rightnodehash = rightnode.recursive_set(db, key, value)?;
                    self.rightnode = rightnodehash;
                } else {
                    let newnode = Node::new_node(db, key, value)?;
                    self.link_right(db, &newnode)?;
                }
            }
        }

        let height = self.get_height(db)?;
        self.write(db)?;
        finalhash = self.get_hash()?;
        if height > 1 {
            finalhash = self.balance(db)?;
        }
        Ok(finalhash)
    }

    pub fn new_node(db: &Db, key: i64, value: String) -> Result<Node> {
        let ret = Node {
            key,
            value,
            leftnode: Bytes::new(),
            rightnode: Bytes::new(),
        };

        ret.write(db)?;

        Ok(ret)
    }
    pub fn get_hash(&self) -> Result<Bytes> {
        let b = bincode::serialize(&self)?;
        Ok(Bytes::from(sha2::Sha256::digest(&b).to_vec()))
    }

    pub fn link_left(&mut self, db: &Db, leftnode: &Node) -> Result<Bytes> {
        self.leftnode = leftnode.get_hash()?;
        let ret = self.write(db)?;
        Ok(ret)
    }

    pub fn link_right(&mut self, db: &Db, rightnode: &Node) -> Result<Bytes> {
        self.rightnode = rightnode.get_hash()?;
        let ret = self.write(db)?;
        Ok(ret)
    }

    pub fn link(&mut self, db: &Db, leftnode: &Node, rightnode: &Node) -> Result<Bytes> {
        self.leftnode = leftnode.get_hash()?;
        self.rightnode = rightnode.get_hash()?;
        let ret = self.write(db)?;
        Ok(ret)
    }

    pub fn auto_link(db: &Db, key: i64, left: i64, right: i64) -> Result<Node> {
        let mut parentnode = Node::new_number_node(db, key)?;
        let leftnode = Node::new_number_node(db, left)?;
        let rightnode = Node::new_number_node(db, right)?;
        let _ = parentnode.link(db, &leftnode, &rightnode)?;
        Ok(parentnode)
    }

    pub fn generate_random_bytes(a: usize) -> Bytes {
        let mut rng = rand::thread_rng();
        let mut bytes = BytesMut::with_capacity(a);
        for _ in 0..a {
            bytes.put_u8(rng.gen());
        }
        bytes.freeze()
    }

    pub fn generate_random_string(a: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut s = String::with_capacity(a);
        for _ in 0..a {
            let alphabet = rng.gen_range(b'A'..=b'Z') as u8;
            s.push(alphabet as char);
        }
        s
    }

    #[allow(dead_code)]
    pub fn new_random_node() -> Node {
        Node {
            key: rand::thread_rng().gen_range(0..100),
            value: Node::generate_random_string(5),
            leftnode: Node::generate_random_bytes(32),
            rightnode: Node::generate_random_bytes(32),
        }
    }

    #[allow(dead_code)]
    pub fn new_random_node_basic() -> Node {
        Node {
            key: rand::thread_rng().gen_range(0..100),
            value: Node::generate_random_string(5),
            leftnode: Bytes::new(),
            rightnode: Bytes::new(),
        }
    }

    // read Node from sled db
    pub fn read(myhash: &Bytes, db: &Db) -> Result<Node> {
        let e = db
            .get(myhash)?
            .ok_or_else(|| anyhow::anyhow!("not found"))?;
        let f = bincode::deserialize::<Node>(&e)?;
        Ok(f)
    }
    // write Node to sled db
    pub fn write(&self, db: &Db) -> Result<Bytes> {
        let b = bincode::serialize(&self)?;
        let myhash = self.get_hash()?;
        db.insert(&myhash, b)?;
        Ok(myhash)
    }

    //      thisnode
    //    l         _r
    //  _ll lr     _rl _rr

    // swap  lr and thisnode
    //           l
    //     _ll            thisnode
    //                 lr      _r
    //                         _rl _rr
    pub fn rotate_right(&self, db: &Db) -> Result<Bytes> {
        println!("node {} rotate_right", self.key);
        let mut thisnode = self.clone();
        let leftnode = Node::read(&thisnode.leftnode, db)?;
        let mut leftrightnodehash = Bytes::new();
        if !leftnode.rightnode.is_empty() {
            leftrightnodehash = Node::read(&leftnode.rightnode, db)?.get_hash()?;
        }

        thisnode.leftnode.clear();
        thisnode.leftnode = leftrightnodehash;
        thisnode.write(db)?;

        let mut newthisnode = leftnode;
        newthisnode.rightnode = thisnode.get_hash()?;
        newthisnode.write(db)?;

        let finalhash = newthisnode.get_hash()?;
        Ok(finalhash)
    }

    //      thisnode
    //    _l           r
    //  _ll _lr     rl _rr

    // swap  rl and thisnode
    //               r
    //    thisnode       _rr
    //     _l     rl
    //  -ll  _lr

    pub fn rotate_left(&self, db: &Db) -> Result<Bytes> {
        println!("node {} rotate_left", self.key);
        let mut thisnode = self.clone();
        let rightnode = Node::read(&thisnode.rightnode, db)?;
        let mut rightleftnodehash = Bytes::new();
        if !rightnode.leftnode.is_empty() {
            rightleftnodehash = Node::read(&rightnode.leftnode, db)?.get_hash()?;
        }
        thisnode.rightnode.clear();
        thisnode.rightnode = rightleftnodehash;
        thisnode.write(db)?;

        let mut newthisnode = rightnode;
        newthisnode.leftnode = thisnode.get_hash()?;
        newthisnode.write(db)?;

        let finalhash = newthisnode.get_hash()?;
        Ok(finalhash)
    }
    pub fn print(&self) {
        println!("{}", self);
    }

    pub fn get_leftnode(&self, db: &Db) -> Result<Node> {
        let leftnode = Node::read(&self.leftnode, db)?;
        Ok(leftnode)
    }
    pub fn get_rightnode(&self, db: &Db) -> Result<Node> {
        let rightnode = Node::read(&self.rightnode, db)?;
        Ok(rightnode)
    }

    pub fn get_height(&self, db: &Db) -> Result<i64> {
        let height = self.recursive_get_height(db)?;
        Ok(height)
    }
    pub fn recursive_get_height(&self, db: &Db) -> Result<i64> {
        if self.leftnode.is_empty() && self.rightnode.is_empty() {
            Ok(0)
        } else if !self.leftnode.is_empty() && self.rightnode.is_empty() {
            let leftnode = self.get_leftnode(db)?;
            Ok(leftnode.recursive_get_height(db)? + 1)
        } else if self.leftnode.is_empty() && !self.rightnode.is_empty() {
            let rightnode = self.get_rightnode(db)?;
            Ok(rightnode.recursive_get_height(db)? + 1)
        } else {
            let leftnode = self.get_leftnode(db)?;
            let rightnode = self.get_rightnode(db)?;
            let leftnodeheight = leftnode.recursive_get_height(db)? + 1;
            let rightnodeheight = rightnode.recursive_get_height(db)? + 1;
            if leftnodeheight > rightnodeheight {
                Ok(leftnodeheight)
            } else {
                Ok(rightnodeheight)
            }
        }
    }

    pub fn print_all(&self, db: &Db) -> Result<()> {
        self.print();
        if !self.leftnode.is_empty() {
            let leftnode = Node::read(&self.leftnode, db)?;
            leftnode.print_all(db)?;
        }
        if !self.rightnode.is_empty() {
            let rightnode = Node::read(&self.rightnode, db)?;
            rightnode.print_all(db)?;
        }

        Ok(())
    }

    pub fn compute_balance(&self, db: &Db) -> Result<i64> {
        let mut leftheight = 0;
        let mut rightheight = 0;
        if self.leftnode.is_empty() {
            leftheight = 0;
        }
        if self.rightnode.is_empty() {
            rightheight = 0;
        }

        if !self.leftnode.is_empty() {
            leftheight = self.get_leftnode(db)?.get_height(db)? + 1;
        }

        if !self.rightnode.is_empty() {
            rightheight = self.get_rightnode(db)?.get_height(db)? + 1;
        }
        let diff = leftheight - rightheight;
        Ok(diff)
    }

    pub fn balance(&mut self, db: &Db) -> Result<Bytes> {
        let mut finalhash = self.get_hash()?;
        let computed_balance = self.compute_balance(db)?;
        println!(
            "balance node {}  computed_balance {}",
            self.key, computed_balance
        );
        if computed_balance > 1 {
            let left_node = self.get_leftnode(db)?;
            let left_balance = left_node.compute_balance(db)?;
            if left_balance >= 0 {
                // left left
                println!("node {} left left", self.key);
                finalhash = self.rotate_right(db)?;
            } else {
                // left right
                println!("node {} left right", self.key);
                self.leftnode = left_node.rotate_left(db)?;
                finalhash = self.rotate_right(db)?;
            }
        } else if computed_balance < -1 {
            let right_node = self.get_rightnode(db)?;
            let right_balance = right_node.compute_balance(db)?;
            if right_balance <= 0 {
                // right right
                println!("node {} right right", self.key);
                finalhash = self.rotate_left(db)?;
            } else {
                // right left
                println!("node {} right left", self.key);
                self.rightnode = right_node.rotate_right(db)?;
                finalhash = self.rotate_left(db)?;
            }
        }
        Ok(finalhash)
    }

    pub fn make_tree(&self, db: &Db) -> Result<StringTreeNode> {
        let mut node = StringTreeNode::new(self.key.to_string());

        if self.leftnode.is_empty() && self.rightnode.is_empty() {
            return Ok(node);
        }

        if !self.rightnode.is_empty() {
            let rightnode = Node::read(&self.rightnode, db)?;
            if let Ok(righttree) = rightnode.make_tree(db) {
                node.push_node(righttree);
            } else {
                node.push("empty".to_string());
            }
        } else {
            node.push("empty".to_string());
        }

        if !self.leftnode.is_empty() {
            let leftnode = Node::read(&self.leftnode, db)?;
            if let Ok(lefttree) = leftnode.make_tree(db) {
                node.push_node(lefttree);
            } else {
                node.push("empty".to_string());
            }
        } else {
            node.push("empty".to_string());
        }

        Ok(node)
    }
}
