#pragma once

#include <iostream>

#ifdef LLAISYS_ENABLE_LOG

#define LLAISYS_LOG_IMPL(level, msg)                                             \
    do {                                                                         \
        std::cout << "[" level "] " << __FILE__ << ":" << __LINE__ << " " << msg \
                  << std::endl;                                                  \
    } while (0)

#define LOG_DEBUG(msg) LLAISYS_LOG_IMPL("DEBUG", msg)
#define LOG_INFO(msg) LLAISYS_LOG_IMPL("INFO", msg)
#define LOG_WARN(msg) LLAISYS_LOG_IMPL("WARN", msg)
#define LOG_ERROR(msg) LLAISYS_LOG_IMPL("ERROR", msg)

#define LOG_TENSOR_DEBUG(label, tensor_ptr)      \
    do {                                         \
        LOG_DEBUG(label);                        \
        if ((tensor_ptr) != nullptr) {           \
            (tensor_ptr)->debug();               \
        } else {                                 \
            LOG_WARN(label << " (null tensor)"); \
        }                                        \
    } while (0)

#define LOG_TENSOR_META(label, tensor_ptr)                                     \
    do {                                                                       \
        if ((tensor_ptr) != nullptr) {                                         \
            LOG_INFO(label << " " << (tensor_ptr)->info()                      \
                           << " contiguous=" << (tensor_ptr)->isContiguous()); \
        } else {                                                               \
            LOG_WARN(label << " (null tensor)");                               \
        }                                                                      \
    } while (0)

#define LOG_TENSOR_META_AT(label, tensor_ptr)                                          \
    do {                                                                               \
        if ((tensor_ptr) != nullptr) {                                                 \
            LLAISYS_LOG_IMPL("INFO",                                                   \
                             label << " " << (tensor_ptr)->info()                      \
                                   << " contiguous=" << (tensor_ptr)->isContiguous()); \
        } else {                                                                       \
            LLAISYS_LOG_IMPL("WARN", label << " (null tensor)");                       \
        }                                                                              \
    } while (0)

#else

#define LOG_DEBUG(msg) \
    do {               \
    } while (0)
#define LOG_INFO(msg) \
    do {              \
    } while (0)
#define LOG_WARN(msg) \
    do {              \
    } while (0)
#define LOG_ERROR(msg) \
    do {               \
    } while (0)
#define LOG_TENSOR_DEBUG(label, tensor_ptr) \
    do {                                    \
    } while (0)
#define LOG_TENSOR_META(label, tensor_ptr) \
    do {                                   \
    } while (0)
#define LOG_TENSOR_META_AT(label, tensor_ptr) \
    do {                                      \
    } while (0)

#endif
