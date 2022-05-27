#include "demo/include/apple.h"
#include "demo/src/main.rs.h"
#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <iostream>
using namespace std;
namespace org {
namespace applestore {

ApplestoreClient::ApplestoreClient()
{}

ApplestoreClient::~ApplestoreClient()
{

}

void ApplestoreClient::onConnected(rust::String data) const
{
  std::cout << "onConnected: " << data.c_str() << std::endl;
}

void ApplestoreClient::onDisconnected(rust::String data) const
{
  std::cout << "onDisconnected: " << data.c_str() << std::endl;
}

std::unique_ptr<ApplestoreClient> new_applestore_client() {
  return std::make_unique<ApplestoreClient>();
}

} // namespace applestore
} // namespace org
