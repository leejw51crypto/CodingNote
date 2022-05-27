use crate::node::Node;
use anyhow::Result;
use bytes::Bytes;
use sled::Db;
use text_trees::StringTreeNode;
use text_trees::TreeFormatting;
use text_trees::FormatCharacters;


#[derive(Debug, Clone)]
pub struct Tree {
    roothash: Bytes,
    db: Db,
}

impl Tree {
    pub fn new() -> Result<Tree> {
        let db = sled::open("iavl.db")?;
        Ok(Tree {
            roothash: Bytes::new(),
            db,
        })
    }
    pub fn read_node(&self, myhash: &Bytes) -> Result<Node> {
        let newnode = Node::read(myhash, &self.db)?;
        Ok(newnode)
    }

    #[allow(dead_code)]
    pub fn write_node(&self, mynode: &Node) -> Result<Bytes> {
        let myhash = mynode.write(&self.db)?;
        Ok(myhash)
    }
    pub fn load(&mut self) -> Result<()> {
        // save roothash to db
        // load Bytes from db with key "roothash"
        let e = self.db.get("roothash")?;
        if let Some(v) = e {
            // create Bytes from v
            self.roothash = Bytes::from(v.to_vec());
        }
        Ok(())
    }
    pub fn save(&self) -> Result<()> {
        // save roothash to db
        self.db.insert("roothash", self.roothash.to_vec())?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_test_for_rotate_right(&mut self) -> Result<()> {
        // open sled db

        if self.roothash.is_empty() {
            let mut a8 = Node::new_number_node(&self.db, 8)?;
            let a6 = Node::auto_link(&self.db, 6, 5, 7)?;
            let a2 = Node::auto_link(&self.db, 2, 1, 3)?;
            let mut a4 = Node::new_number_node(&self.db, 4)?;
            a4.link(&self.db, &a2, &a6)?;
            let a9 = Node::new_number_node(&self.db, 9)?;

            a8.link(&self.db, &a4, &a9)?;
            let finalhash = a8.link(&self.db, &a4, &a9)?;
            self.roothash = finalhash;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_test2(&mut self) -> Result<()> {
        // open sled db

        if self.roothash.is_empty() {
            let mut a2 = Node::new_number_node(&self.db, 2)?;

            let a8 = Node::auto_link(&self.db, 8, 7, 9)?;
            let a4 = Node::auto_link(&self.db, 4, 5, 3)?;
            let mut a6 = Node::new_number_node(&self.db, 6)?;
            a6.link(&self.db, &a4, &a8)?;

            let a1 = Node::new_number_node(&self.db, 1)?;
            let finalhash = a2.link(&self.db, &a1, &a6)?;
            self.roothash = finalhash;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn write_test3(&mut self) -> Result<()> {
        //if self.roothash.is_empty()
        {
            let mut a2 = Node::new_number_node(&self.db, 2)?;

            let a8 = Node::auto_link(&self.db, 8, 7, 9)?;
            let a4 = Node::auto_link(&self.db, 4, 5, 3)?;
            let mut a6 = Node::new_number_node(&self.db, 6)?;
            a6.link(&self.db, &a4, &a8)?;

            let a1 = Node::new_number_node(&self.db, 1)?;
            let finalhash = a2.link(&self.db, &a1, &a6)?;
            self.roothash = finalhash;

            self.roothash = a2.balance(&self.db)?;
        }

        Ok(())
    }

    pub fn get(&mut self, key: i64) -> Result<String> {
        let mut rootnode = self.read_node(&self.roothash)?;
        let value = rootnode.recursive_get(&self.db, key)?;
        Ok(value)
    }
    pub fn set(&mut self, key: i64, value: String) -> Result<()> {
        let newnode = Node::new_node(&self.db, key, value.clone())?;
        if self.roothash.is_empty() {
            self.roothash = newnode.get_hash()?;
        } else {
            let mut rootnode = self.read_node(&self.roothash)?;
            self.roothash = rootnode.recursive_set(&self.db, key, value)?;
        }

        Ok(())
    }

    fn make_tree(&self) -> Result<StringTreeNode> {
        let newroot = self.read_node(&self.roothash)?;
        newroot.make_tree(&self.db)
    }

    pub fn print(&self) -> Result<()> {
        // print self.roothash
        println!("------------------------------------");
        println!("roothash={}", hex::encode(&self.roothash[0..4]));
        let newroot = self.read_node(&self.roothash)?;
        _ = newroot.print_all(&self.db);

        self.show_graph()?;
        Ok(())
    }
    #[allow(dead_code)]
    pub fn write_test(&mut self) -> Result<()> {
        //let mut a2 = Node::new_number_node(&self.db, 2)?;
        self.set(2, "two".to_string())?;
        self.print()?;

        self.set(5, "five".to_string())?;
        self.print()?;

        self.set(20, "twenty".to_string())?;
        self.print()?;

        self.set(1, "one".to_string())?;
        self.print()?;

        self.set(-10, "minus ten".to_string())?;
        self.print()?;

        self.set(30000, "three thousand".to_string())?;
        self.print()?;

        self.set(11, "eleven".to_string())?;
        self.print()?;
        //self.roothash= a2.get_hash()?;

        self.set(15, "fifteen".to_string())?;
        self.print()?;

        self.set(15, "new fifteen".to_string())?;
        self.print()?;

        self.set(5, "new five".to_string())?;
        self.print()?;

        Ok(())
    }
    fn show_graph(&self) -> Result<()> {
        let graph = self.make_tree()?;
        let result=graph.to_string_with_format(
            &TreeFormatting::dir_tree(FormatCharacters::box_chars())
        )?;
        println!("{}",result);
        //println!("{}", graph);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn run(&mut self) -> Result<()> {
        _ = self.load();
        self.write_test()?;
        self.save()?;
        let newroothash = self.roothash.clone();
        let newroot = self.read_node(&newroothash)?;
        let height = newroot.get_height(&self.db)?;
        println!("root={} height={}", hex::encode(&newroothash[0..4]), height);

        let value20 = self.get(20)?;
        println!("value20={}", value20);

        Ok(())
    }
}
