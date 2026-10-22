/* Minimal consumer of the installed C API: prints the library version and the device probe.
 * Exit status 0 whenever the library loads and the probe answers (with or without a GPU). */
#include <mossinfer/mossinfer.h>

#include <stdio.h>
#include <string.h>

int main(void) {
  printf("mossinfer %s (ABI %d, header ABI %d)\n", mi_version(), (int)mi_abi_version(), MI_ABI_VERSION);
  if (mi_abi_version() != MI_ABI_VERSION) {
    fprintf(stderr, "ABI mismatch between header and library\n");
    return 1;
  }

  mi_probe_info info;
  memset(&info, 0, sizeof(info));
  info.struct_size = sizeof(info);
  mi_status st = mi_probe(&info);
  if (st != MI_OK) {
    fprintf(stderr, "mi_probe: %s: %s\n", mi_status_string(st), mi_last_error_message());
    return 1;
  }
  printf("driver CUDA %d, runtime CUDA %d, driver compatible %d, devices %d (%d supported), flavour \"%s\"\n",
         (int)info.driver_cuda_version, (int)info.runtime_cuda_version, (int)info.driver_compatible,
         (int)info.device_count, (int)info.supported_device_count, info.recommended_flavour);
  printf("%s\n", info.message);

  for (int32_t i = 0; i < info.device_count; ++i) {
    mi_device_info dev;
    memset(&dev, 0, sizeof(dev));
    dev.struct_size = sizeof(dev);
    st = mi_probe_device(i, &dev);
    if (st != MI_OK) {
      fprintf(stderr, "mi_probe_device(%d): %s: %s\n", (int)i, mi_status_string(st), mi_last_error_message());
      return 1;
    }
    printf("device %d: %s, sm_%d%d, %llu MiB total, %llu MiB free, supported %d\n", (int)dev.index, dev.name,
           (int)dev.compute_major, (int)dev.compute_minor, (unsigned long long)(dev.total_memory_bytes >> 20),
           (unsigned long long)(dev.free_memory_bytes >> 20), (int)dev.supported);
  }
  return 0;
}
