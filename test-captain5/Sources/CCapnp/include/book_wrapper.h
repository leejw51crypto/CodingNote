#ifndef BOOK_WRAPPER_H
#define BOOK_WRAPPER_H

#include <stddef.h>  // for size_t
#include <stdint.h>  // for uint64_t, int64_t
#include <stdbool.h> // for bool

#ifdef __cplusplus
namespace capnp { class MallocMessageBuilder; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct BookMessage_t {
    void* impl;
};
struct FruitMessage_t {
    void* impl;
};

// Opaque pointer types
typedef struct BookMessage_t* BookMessage;
typedef struct FruitMessage_t* FruitMessage;

// Book functions
BookMessage book_create_book(void);
void book_delete_message(BookMessage msg);
void book_set_title(BookMessage msg, const char* title);
void book_set_author(BookMessage msg, const char* author);
void book_set_pages(BookMessage msg, uint32_t pages);
void book_set_publish_year(BookMessage msg, uint16_t year);
void book_set_is_available(BookMessage msg, bool available);
void book_set_genres(BookMessage msg, const char** genres, size_t count);

const char* book_get_title(BookMessage msg);
const char* book_get_author(BookMessage msg);
uint32_t book_get_pages(BookMessage msg);
uint16_t book_get_publish_year(BookMessage msg);
bool book_get_is_available(BookMessage msg);
size_t book_get_genres_count(BookMessage msg);
const char* book_get_genre(BookMessage msg, size_t index);

// Fruit functions
FruitMessage book_create_fruit(void);
void fruit_delete_message(FruitMessage msg);
void fruit_set_name(FruitMessage msg, const char* name);
void fruit_set_color(FruitMessage msg, const char* color);
void fruit_set_weight_grams(FruitMessage msg, uint32_t weight);
void fruit_set_is_ripe(FruitMessage msg, bool ripe);
void fruit_set_variety(FruitMessage msg, const char* variety);

const char* fruit_get_name(FruitMessage msg);
const char* fruit_get_color(FruitMessage msg);
uint32_t fruit_get_weight_grams(FruitMessage msg);
bool fruit_get_is_ripe(FruitMessage msg);
const char* fruit_get_variety(FruitMessage msg);

// Common serialization functions
unsigned char* book_message_serialize(void* msg, size_t* size);
void book_message_deserialize(void* msg, const unsigned char* data, size_t size);
void book_free_buffer(unsigned char* buffer);

#ifdef __cplusplus
}
#endif

#endif // BOOK_WRAPPER_H 