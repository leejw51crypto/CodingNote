#include "rust/cxx.h"
namespace org {
namespace blobstore {
void test();
int secureStorageWrite(rust::String userkey2, rust::String uservalue2);
rust::String secureStorageRead(rust::String userkey2);
} // namespace blobstore
} // namespace org