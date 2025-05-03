fn count_digits(n: u64) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut count = 0;
    let mut num = n;
    while num > 0 {
        count += 1;
        num /= 10;
    }
    count
}

fn main() {
    // int64 (i64)
    let i64_max = i64::MAX;
    println!("i64 maximum: {}", i64_max);
    println!("i64 maximum digits length: {}", count_digits(i64_max as u64));
    println!("i64 maximum hex: 0x{:X}\n", i64_max);

    // uint64 (u64)
    let u64_max = u64::MAX;
    println!("u64 maximum: {}", u64_max);
    println!("u64 maximum digits length: {}", count_digits(u64_max));
    println!("u64 maximum hex: 0x{:X}\n", u64_max);

    // int32 (i32)
    let i32_max = i32::MAX;
    println!("i32 maximum: {}", i32_max);
    println!("i32 maximum digits length: {}", count_digits(i32_max as u64));
    println!("i32 maximum hex: 0x{:X}\n", i32_max);

    // uint32 (u32)
    let u32_max = u32::MAX;
    println!("u32 maximum: {}", u32_max);
    println!("u32 maximum digits length: {}", count_digits(u32_max as u64));
    println!("u32 maximum hex: 0x{:X}\n", u32_max);
}
