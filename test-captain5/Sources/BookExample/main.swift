import CCapnp
import Foundation

func printBookDetails(_ book: BookMessage!) {
    print("Book Details:")
    print("Title:", String(cString: book_get_title(book)))
    print("Author:", String(cString: book_get_author(book)))
    print("Pages:", book_get_pages(book))
    print("Publish Year:", book_get_publish_year(book))
    print("Is Available:", book_get_is_available(book))
    
    let genreCount = book_get_genres_count(book)
    print("Genres:")
    for i in 0..<genreCount {
        print("  -", String(cString: book_get_genre(book, i)))
    }
}

func printFruitDetails(_ fruit: FruitMessage!) {
    print("\nFruit Details:")
    print("Name:", String(cString: fruit_get_name(fruit)))
    print("Color:", String(cString: fruit_get_color(fruit)))
    print("Weight:", fruit_get_weight_grams(fruit), "grams")
    print("Is Ripe:", fruit_get_is_ripe(fruit))
    print("Variety:", String(cString: fruit_get_variety(fruit)))
}

// Create a book
let book = book_create_book()
defer { book_delete_message(book) }

book_set_title(book, "The Art of Programming")
book_set_author(book, "John Doe")
book_set_pages(book, 500)
book_set_publish_year(book, 2023)
book_set_is_available(book, true)

let genres = ["Programming", "Computer Science"]
let genrePointers = genres.map { strdup($0)! }
defer { genrePointers.forEach { free($0) } }

// Convert array of C strings to UnsafeMutablePointer<UnsafePointer<CChar>?>
let genrePointersPtr = UnsafeMutablePointer<UnsafePointer<CChar>?>.allocate(capacity: genres.count)
defer { genrePointersPtr.deallocate() }

for (i, ptr) in genrePointers.enumerated() {
    genrePointersPtr[i] = UnsafePointer(ptr)
}

book_set_genres(book, genrePointersPtr, genres.count)

// Create a fruit
let fruit = book_create_fruit()
defer { fruit_delete_message(fruit) }

fruit_set_name(fruit, "Apple")
fruit_set_color(fruit, "Red")
fruit_set_weight_grams(fruit, 200)
fruit_set_is_ripe(fruit, true)
fruit_set_variety(fruit, "Honeycrisp")

// Print original messages
print("Original Messages:")
printBookDetails(book)
printFruitDetails(fruit)

// Serialize messages
var size: size_t = 0
let data = book_message_serialize(book, &size)
defer { book_free_buffer(data) }

// Create new messages and deserialize
let newBook = book_create_book()
defer { book_delete_message(newBook) }

// Print the original book again to ensure it's still valid
print("\nOriginal Book (before deserialization):")
printBookDetails(book)

// Perform deserialization
book_message_deserialize(newBook, data, size)

print("\nDeserialized Messages:")
printBookDetails(newBook) 