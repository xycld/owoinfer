// A stand-in server for tests/server/test_shm.py: owns tools/server/shm.h regions and takes commands on stdin, one per
// line, answering one line each (so the test can kill it at any point between them):
//   create <bytes>           a region of <bytes> payload bytes -> "<index> <name> <token hex> <region bytes>"
//   pattern <index> <n>      writes byte i = (i * 7 + 3) % 256 to the first n payload bytes -> "ok"
//   verify <index> <n>       checks the first n payload bytes are i * 13 % 256 (what the test's client writes) -> ok|bad
//   unlink <index>           Region::unlink (the first request naming the region) -> "ok"
//   free <index>             destroys the region -> "ok"
//   pid                      -> the process id
// `shm_helper sweep` instead runs sweep_stale() once (a new server's start) and prints each removed name.
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "shm.h"

#ifdef _WIN32
#include <process.h>
#define MI_GETPID _getpid
#else
#include <unistd.h>
#define MI_GETPID getpid
#endif

namespace shm = mi::server::shm;

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "sweep") {
    for (const std::string& name : shm::sweep_stale()) std::printf("%s\n", name.c_str());
    return 0;
  }
  std::vector<std::unique_ptr<shm::Region>> regions;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    std::size_t index = 0, n = 0;
    try {
      if (cmd == "create") {
        in >> n;
        regions.push_back(std::make_unique<shm::Region>(shm::Region::create(n)));
        const shm::Region& r = *regions.back();
        std::printf("%zu %s %s %zu\n", regions.size() - 1, r.name().c_str(), r.token().hex().c_str(), r.bytes());
      } else if (cmd == "pattern" || cmd == "verify") {
        in >> index >> n;
        std::byte* p = regions.at(index)->payload();
        bool ok = n <= regions.at(index)->payload_bytes();
        for (std::size_t i = 0; ok && i < n; ++i) {
          if (cmd == "pattern") {
            p[i] = static_cast<std::byte>((i * 7 + 3) % 256);
          } else {
            ok = p[i] == static_cast<std::byte>(i * 13 % 256);
          }
        }
        std::printf("%s\n", ok ? "ok" : "bad");
      } else if (cmd == "unlink") {
        in >> index;
        regions.at(index)->unlink();
        std::printf("ok\n");
      } else if (cmd == "free") {
        in >> index;
        regions.at(index).reset();
        std::printf("ok\n");
      } else if (cmd == "pid") {
        std::printf("%d\n", static_cast<int>(MI_GETPID()));
      } else {
        std::printf("error unknown command\n");
      }
    } catch (const std::exception& e) {
      std::printf("error %s\n", e.what());
    }
    std::fflush(stdout);
  }
  return 0;
}
