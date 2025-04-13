import { Message } from 'capnp-es';
import { Book, Fruit, Message as MessageType, BookList } from './proto/book.js';

// Example of creating a new Book message
function createAndSerializeBook() {
    const message = new Message();
    const book = message.initRoot(Book);

    // Set the book fields
    book.title = "The TypeScript Guide";
    book.author = "John Developer";
    book.pages = 300;
    book.publishYear = 2024;
    const genres = book._initGenres(2);
    genres.set(0, "Programming");
    genres.set(1, "Technology");
    book.isAvailable = true;

    // Serialize the book message
    const buffer = message.toPackedArrayBuffer();
    return buffer;
}

// Example of creating a new Fruit message
function createAndSerializeFruit() {
    const message = new Message();
    const fruit = message.initRoot(Fruit);

    // Set the fruit fields
    fruit.name = "Apple";
    fruit.color = "Red";
    fruit.weightGrams = 200;
    fruit.isRipe = true;
    fruit.variety = "Honeycrisp";

    // Serialize the fruit message
    const buffer = message.toPackedArrayBuffer();
    return buffer;
}

// Example of creating a BookList
function createAndSerializeBookList() {
    const message = new Message();
    const bookList = message.initRoot(BookList);
    const books = bookList._initBooks(2);

    // First book
    const book1 = books.get(0);
    book1.title = "Book One";
    book1.author = "Author One";
    book1.pages = 200;
    book1.publishYear = 2023;
    const genres1 = book1._initGenres(1);
    genres1.set(0, "Fiction");
    book1.isAvailable = true;

    // Second book
    const book2 = books.get(1);
    book2.title = "Book Two";
    book2.author = "Author Two";
    book2.pages = 150;
    book2.publishYear = 2024;
    const genres2 = book2._initGenres(1);
    genres2.set(0, "Non-fiction");
    book2.isAvailable = false;

    return message.toPackedArrayBuffer();
}

// Deserialize and display book
function deserializeAndDisplayBook(buffer: ArrayBuffer) {
    const message = new Message(buffer, true);
    const book = message.getRoot(Book);

    console.log('\n=== Book Details ===');
    console.log('Title:', book.title);
    console.log('Author:', book.author);
    console.log('Pages:', book.pages);
    console.log('Publish Year:', book.publishYear);
    console.log('Genres:', Array.from({ length: book.genres.length }, (_, i) => book.genres.get(i)));
    console.log('Is Available:', book.isAvailable);
}

// Deserialize and display fruit
function deserializeAndDisplayFruit(buffer: ArrayBuffer) {
    const message = new Message(buffer, true);
    const fruit = message.getRoot(Fruit);

    console.log('\n=== Fruit Details ===');
    console.log('Name:', fruit.name);
    console.log('Color:', fruit.color);
    console.log('Weight (g):', fruit.weightGrams);
    console.log('Is Ripe:', fruit.isRipe);
    console.log('Variety:', fruit.variety);
}

// Deserialize and display book list
function deserializeAndDisplayBookList(buffer: ArrayBuffer) {
    const message = new Message(buffer, true);
    const bookList = message.getRoot(BookList);

    console.log('\n=== Book List ===');
    for (let i = 0; i < bookList.books.length; i++) {
        const book = bookList.books.get(i);
        console.log(`\nBook ${i + 1}:`);
        console.log('Title:', book.title);
        console.log('Author:', book.author);
        console.log('Pages:', book.pages);
        console.log('Publish Year:', book.publishYear);
        console.log('Genres:', Array.from({ length: book.genres.length }, (_, i) => book.genres.get(i)));
        console.log('Is Available:', book.isAvailable);
    }
}

// Run examples
console.log('Running Cap\'n Proto TypeScript Examples...');

// Book example
const bookBuffer = createAndSerializeBook();
deserializeAndDisplayBook(bookBuffer);

// Fruit example
const fruitBuffer = createAndSerializeFruit();
deserializeAndDisplayFruit(fruitBuffer);

// BookList example
const bookListBuffer = createAndSerializeBookList();
deserializeAndDisplayBookList(bookListBuffer); 