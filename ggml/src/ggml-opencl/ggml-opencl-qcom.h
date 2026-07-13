#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifndef CL_MEM_EXT_HOST_PTR_QCOM
typedef struct _cl_mem_ext_host_ptr {
    cl_uint allocation_type;
    cl_uint host_cache_policy;
} cl_mem_ext_host_ptr;

#define CL_MEM_EXT_HOST_PTR_QCOM (1ULL << 29)
#endif

#ifndef CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
#define CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM 0x40A0
#endif

#ifndef CL_DEVICE_PAGE_SIZE_QCOM
#define CL_DEVICE_PAGE_SIZE_QCOM 0x40A1
#endif

#ifndef CL_MEM_HOST_UNCACHED_QCOM
#define CL_MEM_HOST_UNCACHED_QCOM 0x40A4
#endif

#ifndef CL_MEM_DMABUF_HOST_PTR_QCOM
#define CL_MEM_DMABUF_HOST_PTR_QCOM 0x411D

typedef struct _cl_mem_dmabuf_host_ptr {
    cl_mem_ext_host_ptr ext_host_ptr;
    int dmabuf_filedesc;
    void * dmabuf_hostptr;
} cl_mem_dmabuf_host_ptr;
#endif
