#pragma once
#include "rust/cxx.h"
#include <memory>

namespace org {
namespace applestore {
class ApplestoreClient {
public:
  ApplestoreClient();
  virtual ~ApplestoreClient();
  void onConnected(rust::String data) const;
  void onDisconnected(rust::String data) const;
};

std::unique_ptr<ApplestoreClient> new_applestore_client();

} // namespace blobstore
} // namespace org
