use rand::prelude::*;

fn pickword<'a>(words: &'a [&str]) -> &'a str {
    let mut rng = rand::thread_rng();
    let index = rng.gen_range(0.. words.len());
    words[index]
}

fn main() {
    // make 10 word list
    let words = ["apple", "banana", "orange", "grape", "melon", "lemon", "lime", "peach", "pear", "plum"];
    // pick word from words
    let word = pickword(&words);
    // display word
    println!("{}", word);
    
}
