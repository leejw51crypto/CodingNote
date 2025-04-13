#include "include/book_wrapper.h"
#include "book.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>
#include <iostream>
#include <cstring>
#include <memory>

namespace {
// Internal structure definitions
struct BookMessageImpl {
    std::unique_ptr<::capnp::MallocMessageBuilder> message;
    std::unique_ptr<Book::Builder> builder;
    
    BookMessageImpl() {
        message = std::make_unique<::capnp::MallocMessageBuilder>();
        builder = std::make_unique<Book::Builder>(message->initRoot<Book>());
    }
    
    ~BookMessageImpl() = default;
};

struct FruitMessageImpl {
    std::unique_ptr<::capnp::MallocMessageBuilder> message;
    std::unique_ptr<Fruit::Builder> builder;
    
    FruitMessageImpl() {
        message = std::make_unique<::capnp::MallocMessageBuilder>();
        builder = std::make_unique<Fruit::Builder>(message->initRoot<Fruit>());
    }
    
    ~FruitMessageImpl() = default;
};
}

// Book implementation
BookMessage book_create_book() {
    auto* msg = new BookMessage_t;
    msg->impl = new BookMessageImpl();
    return msg;
}

void book_delete_message(BookMessage msg) {
    if (msg) {
        delete static_cast<BookMessageImpl*>(msg->impl);
        delete msg;
    }
}

void book_set_title(BookMessage msg, const char* title) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    impl->builder->setTitle(title);
}

void book_set_author(BookMessage msg, const char* author) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    impl->builder->setAuthor(author);
}

void book_set_pages(BookMessage msg, uint32_t pages) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    impl->builder->setPages(pages);
}

void book_set_publish_year(BookMessage msg, uint16_t year) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    impl->builder->setPublishYear(year);
}

void book_set_is_available(BookMessage msg, bool available) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    impl->builder->setIsAvailable(available);
}

void book_set_genres(BookMessage msg, const char** genres, size_t count) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    auto genresBuilder = impl->builder->initGenres(count);
    for (size_t i = 0; i < count; i++) {
        genresBuilder.set(i, genres[i]);
    }
}

const char* book_get_title(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getTitle().cStr();
}

const char* book_get_author(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getAuthor().cStr();
}

uint32_t book_get_pages(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getPages();
}

uint16_t book_get_publish_year(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getPublishYear();
}

bool book_get_is_available(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getIsAvailable();
}

size_t book_get_genres_count(BookMessage msg) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getGenres().size();
}

const char* book_get_genre(BookMessage msg, size_t index) {
    auto* impl = static_cast<BookMessageImpl*>(msg->impl);
    return impl->builder->getGenres()[index].cStr();
}

// Fruit implementation
FruitMessage book_create_fruit() {
    auto* msg = new FruitMessage_t;
    msg->impl = new FruitMessageImpl();
    return msg;
}

void fruit_delete_message(FruitMessage msg) {
    if (msg) {
        delete static_cast<FruitMessageImpl*>(msg->impl);
        delete msg;
    }
}

void fruit_set_name(FruitMessage msg, const char* name) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    impl->builder->setName(name);
}

void fruit_set_color(FruitMessage msg, const char* color) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    impl->builder->setColor(color);
}

void fruit_set_weight_grams(FruitMessage msg, uint32_t weight) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    impl->builder->setWeightGrams(weight);
}

void fruit_set_is_ripe(FruitMessage msg, bool ripe) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    impl->builder->setIsRipe(ripe);
}

void fruit_set_variety(FruitMessage msg, const char* variety) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    impl->builder->setVariety(variety);
}

const char* fruit_get_name(FruitMessage msg) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    return impl->builder->getName().cStr();
}

const char* fruit_get_color(FruitMessage msg) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    return impl->builder->getColor().cStr();
}

uint32_t fruit_get_weight_grams(FruitMessage msg) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    return impl->builder->getWeightGrams();
}

bool fruit_get_is_ripe(FruitMessage msg) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    return impl->builder->getIsRipe();
}

const char* fruit_get_variety(FruitMessage msg) {
    auto* impl = static_cast<FruitMessageImpl*>(msg->impl);
    return impl->builder->getVariety().cStr();
}

// Common serialization functions
unsigned char* book_message_serialize(void* msg, size_t* size) {
    if (!msg || !size) return nullptr;

    ::capnp::MallocMessageBuilder* message = nullptr;
    if (auto* book = static_cast<BookMessage_t*>(msg)) {
        auto* impl = static_cast<BookMessageImpl*>(book->impl);
        message = impl->message.get();
    } else if (auto* fruit = static_cast<FruitMessage_t*>(msg)) {
        auto* impl = static_cast<FruitMessageImpl*>(fruit->impl);
        message = impl->message.get();
    } else {
        return nullptr;
    }

    kj::Array<capnp::word> serialized = messageToFlatArray(*message);
    *size = serialized.size() * sizeof(capnp::word);
    unsigned char* buffer = new unsigned char[*size];
    memcpy(buffer, serialized.begin(), *size);
    return buffer;
}

void book_message_deserialize(void* msg, const unsigned char* data, size_t size) {
    if (!msg || !data) return;

    try {
        capnp::FlatArrayMessageReader reader(kj::ArrayPtr<const capnp::word>(
            reinterpret_cast<const capnp::word*>(data),
            size / sizeof(capnp::word)));

        if (auto* book = static_cast<BookMessage_t*>(msg)) {
            auto* impl = static_cast<BookMessageImpl*>(book->impl);
            auto root = reader.getRoot<Book>();
            
            // Create new objects
            impl->message = std::make_unique<::capnp::MallocMessageBuilder>();
            auto newRoot = impl->message->initRoot<Book>();
            
            // Copy all fields
            newRoot.setTitle(root.getTitle());
            newRoot.setAuthor(root.getAuthor());
            newRoot.setPages(root.getPages());
            newRoot.setPublishYear(root.getPublishYear());
            newRoot.setIsAvailable(root.getIsAvailable());
            
            auto genres = root.getGenres();
            auto newGenres = newRoot.initGenres(genres.size());
            for (size_t i = 0; i < genres.size(); i++) {
                newGenres.set(i, genres[i]);
            }
            
            // Update builder
            impl->builder = std::make_unique<Book::Builder>(newRoot);
            
        } else if (auto* fruit = static_cast<FruitMessage_t*>(msg)) {
            auto* impl = static_cast<FruitMessageImpl*>(fruit->impl);
            auto root = reader.getRoot<Fruit>();
            
            // Create new objects
            impl->message = std::make_unique<::capnp::MallocMessageBuilder>();
            auto newRoot = impl->message->initRoot<Fruit>();
            
            // Copy all fields
            newRoot.setName(root.getName());
            newRoot.setColor(root.getColor());
            newRoot.setWeightGrams(root.getWeightGrams());
            newRoot.setIsRipe(root.getIsRipe());
            newRoot.setVariety(root.getVariety());
            
            // Update builder
            impl->builder = std::make_unique<Fruit::Builder>(newRoot);
        }
    } catch (const kj::Exception& e) {
        std::cerr << "Error during deserialization: " << e.getDescription().cStr() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during deserialization: " << e.what() << std::endl;
    }
}

void book_free_buffer(unsigned char* buffer) {
    delete[] buffer;
} 