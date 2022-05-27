fn change(mut arr: &mut [i32; 5]) {
    arr[0] = 100;
    arr[4] = 100;
}

fn change2(mut arr: Vec<i32>) {
    arr[0] = 200;
    arr[4] = 200;
}

fn change3(mut arr: [i32; 5]) {
    arr[0] = 100;
    arr[4] = 100;
}

fn main2() {
    {
        let mut arr = [0; 5];
        println!("{:?}", arr);
        change3(arr);
        // print arr
        println!("{:?}", arr);
    }

    {
        let mut arr = [0; 5];
        println!("{:?}", arr);
        change(&mut arr);
        // print arr
        println!("{:?}", arr);
    }

    // make Vec<i32> of 5
    let mut vec = vec![0; 5];
    println!("{:?}", vec);
    change2(vec);
    // print vec
    //  println!("{:?}", vec);
}

fn print2(arr: &[String]) {
    println!("{:?}", arr);
}

fn main()
{
    // make string Vec with 5 elements with data
    let mut vec = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "e".to_string()];
    print2(&vec);
    
}