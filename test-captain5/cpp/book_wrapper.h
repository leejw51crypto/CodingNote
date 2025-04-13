#ifndef book_wrapper_h
#define book_wrapper_h

#include <capnp/message.h>
#include <capnp/serialize.h>
#include "book.capnp.h"

class BookWrapper {
public:
    static void* createBook(const char* title, const char* author, uint32_t pages, 
                          uint16_t publishYear, bool isAvailable,
                          const char* const* genres, size_t genresCount);
    
    static void* createFruit(const char* name, const char* color, uint32_t weightGrams,
                           bool isRipe, const char* variety);
    
    static void deleteMessage(void* message);
    
    static void* serializeMessage(void* message, size_t* size);
    static void* deserializeMessage(const void* data, size_t size);
    
    // Book getters
    static const char* getBookTitle(void* message);
    static const char* getBookAuthor(void* message);
    static uint32_t getBookPages(void* message);
    static uint16_t getBookPublishYear(void* message);
    static bool getBookIsAvailable(void* message);
    static size_t getBookGenresCount(void* message);
    static const char* getBookGenre(void* message, size_t index);
    
    // Fruit getters
    static const char* getFruitName(void* message);
    static const char* getFruitColor(void* message);
    static uint32_t getFruitWeightGrams(void* message);
    static bool getFruitIsRipe(void* message);
    static const char* getFruitVariety(void* message);
};

#endif /* book_wrapper_h */ 