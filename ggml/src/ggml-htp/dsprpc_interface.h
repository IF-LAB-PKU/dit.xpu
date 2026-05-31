// 声明 Qualcomm FastRPC / DSPRPC 底层 API（rpcmem_init, rpcmem_alloc, fastrpc_mmap 等），
// 定义 DSP 域 ID 常量（ADSP / MDSP / CDSP 等）、堆 ID 以及 mmap 标志位。
// 这些是 HTP 后端与 Qualcomm DSP 通信所需的系统级原语。

#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ADSP_DOMAIN_ID  0
#define MDSP_DOMAIN_ID  1
#define SDSP_DOMAIN_ID  2
#define CDSP_DOMAIN_ID  3
#define CDSP1_DOMAIN_ID 4

void   rpcmem_init(void);
void   rpcmem_deinit(void);
void * rpcmem_alloc(int heap_id, uint32_t flags, int size);
void   rpcmem_free(void * p);
int    rpcmem_to_fd(void * p);

#define RPCMEM_FLAG_UNCACHED 0
#define RPCMEM_FLAG_CACHED   1

enum rpc_heap_ids {
    RPCMEM_HEAP_ID_SECURE = 9,
    RPCMEM_HEAP_ID_CONTIG = 22,
    RPCMEM_HEAP_ID_SYSTEM = 25,
};

enum fastrpc_map_flags {
    FASTRPC_MAP_STATIC,
    FASTRPC_MAP_RESERVED,
    FASTRPC_MAP_FD,
    FASTRPC_MAP_FD_DELAYED,
    FASTRPC_MAP_RESERVED_4,
    FASTRPC_MAP_RESERVED_5,
    FASTRPC_MAP_RESERVED_6,
    FASTRPC_MAP_RESERVED_7,
    FASTRPC_MAP_RESERVED_8,
    FASTRPC_MAP_RESERVED_9,
    FASTRPC_MAP_RESERVED_10,
    FASTRPC_MAP_RESERVED_11,
    FASTRPC_MAP_RESERVED_12,
    FASTRPC_MAP_RESERVED_13,
    FASTRPC_MAP_RESERVED_14,
    FASTRPC_MAP_RESERVED_15,
    FASTRPC_MAP_FD_NOMAP,
};

int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags);
int fastrpc_munmap(int domain, int fd, void * addr, size_t length);

bool dsprpc_is_available(void);
const char * dsprpc_last_error(void);

#ifdef __cplusplus
}
#endif
