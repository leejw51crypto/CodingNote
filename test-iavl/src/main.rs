mod node;
mod tree;
use anyhow::Result;
use text_trees::StringTreeNode;

#[allow(dead_code)]
fn main2() -> Result<()> {
    //let a= make_tree();
    // println!("{}", a);
    let mut m = tree::Tree::new()?;
    m.run()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut m = tree::Tree::new()?;
    // loop 
    loop {
        // read int64 from stdin

        let mut input = String::new();
        
        println!("enter q to quit");
        println!("enter number");
        
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        // if input is q, break
        if input == "q" {
            break;
        }
        let input = input.parse::<i64>()?;

        m.set(input, input.to_string())?;
        m.print()?;
    }
    Ok(())
}


#[allow(dead_code)]
fn make_tree() -> StringTreeNode {
    StringTreeNode::with_child_nodes(
        "root".to_string(),
        vec![
            "Uncle".into(),
            StringTreeNode::with_child_nodes(
                "Parent".to_string(),
                vec![
                    StringTreeNode::with_children(
                        "Child 1".to_string(),
                        vec!["Grand Child 1".into()].into_iter(),
                    ),
                    StringTreeNode::with_child_nodes(
                        "Child 2".to_string(),
                        vec![StringTreeNode::with_child_nodes(
                            "Grand Child 2".to_string(),
                            vec![StringTreeNode::with_children(
                                "Great Grand Child 2".to_string(),
                                vec!["Great Great Grand Child 2".to_string()].into_iter(),
                            )]
                            .into_iter(),
                        )]
                        .into_iter(),
                    ),
                ]
                .into_iter(),
            ),
            StringTreeNode::with_children(
                "Aunt".to_string(),
                vec!["Child 3".to_string()].into_iter(),
            ),
        ]
        .into_iter(),
    )
}
