#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {
class Storage {
private:
    std::byte *_memory; // 指向实际内存的指针
    size_t _size;       // 内存大小
    Runtime &_runtime;  // 关联的运行时
    bool _is_host;      // 是否是主机内存
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

}; // namespace llaisys::core
