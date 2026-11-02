#include "vk_api.h"

#include <mutex>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define MI_VK_DEFINE(name) PFN_##name name = nullptr;
MI_VK_FUNCTIONS(MI_VK_DEFINE)
#undef MI_VK_DEFINE

namespace mi::vk {
namespace {

void* open_loader() {
#if defined(_WIN32)
  // From the system directory only, as the Vulkan loader's documentation asks.
  return reinterpret_cast<void*>(LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#elif defined(__APPLE__)
  void* h = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
  return h != nullptr ? h : dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
#else
  void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  return h != nullptr ? h : dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* library, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

}  // namespace

bool load_vulkan(std::string* error) {
  static std::once_flag once;
  static std::string failure;
  std::call_once(once, [] {
    void* library = open_loader();
    if (library == nullptr) {
      failure = "no Vulkan loader found (libvulkan.so.1 / vulkan-1.dll): install the GPU's Vulkan driver";
      return;
    }
    std::string missing;
#define MI_VK_LOAD(name)                                                   \
  name = reinterpret_cast<PFN_##name>(symbol(library, #name));             \
  if (name == nullptr) missing += std::string(missing.empty() ? "" : ", ") + #name;
    MI_VK_FUNCTIONS(MI_VK_LOAD)
#undef MI_VK_LOAD
    if (!missing.empty()) failure = "the Vulkan loader lacks " + missing + " (it is older than Vulkan 1.2)";
  });
  if (!failure.empty() && error != nullptr) *error = failure;
  return failure.empty();
}

}  // namespace mi::vk
