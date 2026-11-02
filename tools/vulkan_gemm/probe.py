"""Lists a Vulkan device's cooperative-matrix shapes and memory heaps, using only vulkan-1.dll.

The C++ tool needs Vulkan headers and an import library; this script needs neither (ctypes only), so it works on a
machine with just the GPU driver installed. It is how the capability table in docs/vulkan-backend-research.md was
produced.

    py -3 probe.py
"""
import ctypes as C
import sys

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR = 1000506001
VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV = 1000249001

COMPONENT = {
    0: "f16", 1: "f32", 2: "f64", 3: "s8", 4: "s16", 5: "s32", 6: "s64",
    7: "u8", 8: "u16", 9: "u32", 10: "u64", 1000141000: "bf16",
}
SCOPE = {1: "device", 2: "workgroup", 3: "subgroup"}


class VkApplicationInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_int32), ("pNext", C.c_void_p), ("pApplicationName", C.c_char_p),
        ("applicationVersion", C.c_uint32), ("pEngineName", C.c_char_p), ("engineVersion", C.c_uint32),
        ("apiVersion", C.c_uint32),
    ]


class VkInstanceCreateInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_int32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
        ("pApplicationInfo", C.POINTER(VkApplicationInfo)), ("enabledLayerCount", C.c_uint32),
        ("ppEnabledLayerNames", C.c_void_p), ("enabledExtensionCount", C.c_uint32),
        ("ppEnabledExtensionNames", C.c_void_p),
    ]


class VkCooperativeMatrixProperties(C.Structure):
    _fields_ = [
        ("sType", C.c_int32), ("pNext", C.c_void_p),
        ("MSize", C.c_uint32), ("NSize", C.c_uint32), ("KSize", C.c_uint32),
        ("AType", C.c_int32), ("BType", C.c_int32), ("CType", C.c_int32), ("ResultType", C.c_int32),
        ("saturatingAccumulation", C.c_uint32), ("scope", C.c_int32),
    ]


class VkMemoryType(C.Structure):
    _fields_ = [("propertyFlags", C.c_uint32), ("heapIndex", C.c_uint32)]


class VkMemoryHeap(C.Structure):
    _fields_ = [("size", C.c_uint64), ("flags", C.c_uint32)]


class VkPhysicalDeviceMemoryProperties(C.Structure):
    _fields_ = [
        ("memoryTypeCount", C.c_uint32), ("memoryTypes", VkMemoryType * 32),
        ("memoryHeapCount", C.c_uint32), ("memoryHeaps", VkMemoryHeap * 16),
    ]


def main():
    try:
        vk = C.WinDLL("vulkan-1.dll")
    except OSError:
        vk = C.CDLL("libvulkan.so.1")
    vk.vkGetInstanceProcAddr.restype = C.c_void_p
    vk.vkGetInstanceProcAddr.argtypes = [C.c_void_p, C.c_char_p]
    gipa = vk.vkGetInstanceProcAddr

    def proc(instance, name, restype, argtypes):
        address = gipa(instance, name.encode())
        return C.CFUNCTYPE(restype, *argtypes)(address) if address else None

    app = VkApplicationInfo(VK_STRUCTURE_TYPE_APPLICATION_INFO, None, b"mossinfer-probe", 1, b"probe", 1, (1 << 22) | (3 << 12))
    info = VkInstanceCreateInfo(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, None, 0, C.pointer(app), 0, None, 0, None)
    instance = C.c_void_p()
    if proc(None, "vkCreateInstance", C.c_int32,
            [C.POINTER(VkInstanceCreateInfo), C.c_void_p, C.POINTER(C.c_void_p)])(C.byref(info), None,
                                                                                C.byref(instance)) != VK_SUCCESS:
        print("vkCreateInstance failed")
        return 1

    enumerate_devices = proc(instance, "vkEnumeratePhysicalDevices", C.c_int32,
                             [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_void_p)])
    count = C.c_uint32(0)
    enumerate_devices(instance, C.byref(count), None)
    devices = (C.c_void_p * count.value)()
    enumerate_devices(instance, C.byref(count), devices)
    get_memory = proc(instance, "vkGetPhysicalDeviceMemoryProperties", None,
                      [C.c_void_p, C.POINTER(VkPhysicalDeviceMemoryProperties)])
    print(f"physical devices: {count.value}")

    for index in range(count.value):
        device = devices[index]
        print(f"\ndevice {index}:")
        # The KHR query is authoritative on current drivers; the NV one is only a fallback for older ones (its enums
        # are narrower, so its properties are not decoded here).
        khr_shapes = 0
        for name, s_type in (("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR",
                              VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR),
                             ("vkGetPhysicalDeviceCooperativeMatrixPropertiesNV",
                              VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV)):
            if name.endswith("NV") and khr_shapes:
                continue
            query = proc(instance, name, C.c_int32,
                         [C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(VkCooperativeMatrixProperties)])
            if query is None:
                continue
            n = C.c_uint32(0)
            result = query(device, C.byref(n), None)
            if result != VK_SUCCESS or n.value == 0:
                continue
            shapes = (VkCooperativeMatrixProperties * n.value)()
            for shape in shapes:
                shape.sType = s_type
            if query(device, C.byref(n), shapes) != VK_SUCCESS:
                continue
            if name.endswith("KHR"):
                khr_shapes = n.value
            print(f"  {name}: {n.value} shapes")
            for shape in shapes[:n.value]:
                print(f"    {shape.MSize}x{shape.NSize}x{shape.KSize} "
                      f"A={COMPONENT.get(shape.AType, shape.AType)} B={COMPONENT.get(shape.BType, shape.BType)} "
                      f"C={COMPONENT.get(shape.CType, shape.CType)} R={COMPONENT.get(shape.ResultType, shape.ResultType)} "
                      f"sat={shape.saturatingAccumulation} scope={SCOPE.get(shape.scope, shape.scope)}")
        if get_memory is not None:
            memory = VkPhysicalDeviceMemoryProperties()
            get_memory(device, C.byref(memory))
            for heap in range(memory.memoryHeapCount):
                info = memory.memoryHeaps[heap]
                kind = " (device local)" if info.flags & 1 else ""
                print(f"  heap {heap}: {info.size / (1024 * 1024):.0f} MiB{kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
