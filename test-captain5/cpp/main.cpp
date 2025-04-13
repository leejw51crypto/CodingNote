#include <capnp/message.h>
#include <capnp/serialize.h>
#include <iostream>
#include <vector>
#include "proto/book.capnp.h"

// Helper function to print a Book
void printBook(const Book::Reader& book) {
    std::cout << "Book:" << std::endl;
    std::cout << "  Title: " << book.getTitle().cStr() << std::endl;
    std::cout << "  Author: " << book.getAuthor().cStr() << std::endl;
    std::cout << "  Pages: " << book.getPages() << std::endl;
    std::cout << "  Publish Year: " << book.getPublishYear() << std::endl;
    std::cout << "  Available: " << (book.getIsAvailable() ? "yes" : "no") << std::endl;
    std::cout << "  Genres: ";
    for (auto genre : book.getGenres()) {
        std::cout << genre.cStr() << ", ";
    }
    std::cout << std::endl;
}

// Helper function to print a Fruit
void printFruit(const Fruit::Reader& fruit) {
    std::cout << "Fruit:" << std::endl;
    std::cout << "  Name: " << fruit.getName().cStr() << std::endl;
    std::cout << "  Color: " << fruit.getColor().cStr() << std::endl;
    std::cout << "  Weight: " << fruit.getWeightGrams() << "g" << std::endl;
    std::cout << "  Ripe: " << (fruit.getIsRipe() ? "yes" : "no") << std::endl;
    std::cout << "  Variety: " << fruit.getVariety().cStr() << std::endl;
}

int main() {
    try {
        std::cout << "Cap'n Proto Serialization Example" << std::endl;
        std::cout << "================================" << std::endl;

        // Create a book message
        capnp::MallocMessageBuilder bookMessage;
        Book::Builder book = bookMessage.initRoot<Book>();
        book.setTitle("The Art of Programming");
        book.setAuthor("John Doe");
        book.setPages(500);
        book.setPublishYear(2024);
        book.setIsAvailable(true);
        
        auto genres = book.initGenres(2);
        genres.set(0, "Programming");
        genres.set(1, "Computer Science");

        // Serialize the book
        kj::Array<capnp::word> bookWords = messageToFlatArray(bookMessage);
        kj::ArrayPtr<const char> bookBytes = bookWords.asChars();
        
        // Create a fruit message
        capnp::MallocMessageBuilder fruitMessage;
        Fruit::Builder fruit = fruitMessage.initRoot<Fruit>();
        fruit.setName("Apple");
        fruit.setColor("Red");
        fruit.setWeightGrams(200);
        fruit.setIsRipe(true);
        fruit.setVariety("Honeycrisp");

        // Serialize the fruit
        kj::Array<capnp::word> fruitWords = messageToFlatArray(fruitMessage);
        kj::ArrayPtr<const char> fruitBytes = fruitWords.asChars();

        std::cout << "\nOriginal Messages:" << std::endl;
        std::cout << "-------------------" << std::endl;
        printBook(book.asReader());
        std::cout << std::endl;
        printFruit(fruit.asReader());

        // Deserialize and verify the book
        std::cout << "\nDeserialized Messages:" << std::endl;
        std::cout << "----------------------" << std::endl;
        
        capnp::FlatArrayMessageReader bookReader(bookWords.asPtr());
        auto deserializedBook = bookReader.getRoot<Book>();
        printBook(deserializedBook);
        
        std::cout << std::endl;
        
        capnp::FlatArrayMessageReader fruitReader(fruitWords.asPtr());
        auto deserializedFruit = fruitReader.getRoot<Fruit>();
        printFruit(deserializedFruit);

        std::cout << "\nSerialization successful!" << std::endl;
        return 0;
    } catch (const kj::Exception& e) {
        std::cerr << "Error: " << e.getDescription().cStr() << std::endl;
        return 1;
    }
} 