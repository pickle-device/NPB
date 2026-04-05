/*
 * pickle_cg_hooks.cpp
 *
 * C++ hooks for Pickle Prefetcher integration with NPB CG benchmark.
 * These functions have C linkage so they can be called from Fortran 90
 * via iso_c_binding.
 *
 * The CG benchmark performs Conjugate Gradient on a sparse matrix stored
 * in CSR format. The dominant cost is the SpMV kernel:
 *
 *   do k = rowstr(j), rowstr(j+1)-1
 *       suml = suml + a(k) * p(colidx(k))
 *   enddo
 *
 * The indirect access pattern  p( colidx(k) )  is structurally identical
 * to the graph-traversal pattern  property( neighbors(k) ):
 *
 *   rowstr  ≡  out_index      (row pointers,    Ranged / Pointer)
 *   colidx  ≡  out_neighbors  (column indices,  SingleElement / Index)
 *   p / z   ≡  node property  (target vector,   SingleElement / Index)
 *
 * Two SpMV kernels are registered because CG multiplies by A twice per
 * outer iteration — once with vector p (25× inner CG iterations) and
 * once with vector z (1× for the residual norm).
 *
 * Fortran 1-based indexing
 * ========================
 * The Pickle device assumes 0-based address arithmetic internally:
 *
 *     address = vaddr_start + index × element_size
 *
 * Fortran stores arr(V) at  &arr(1) + (V-1) × esz.  To make the
 * device's 0-based formula produce the correct address when it receives
 * a 1-based index V from rowstr or colidx, we set:
 *
 *     vaddr_start  =  &arr(1) − esz          ("virtual element 0")
 *
 * so that  vaddr_start + V × esz  =  &arr(V).
 *
 * For the row hint value itself we convert Fortran's 1-based j to
 * 0-based (j − 1) on the Fortran side before writing to UCPage.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <memory>

#if ENABLE_GEM5==1
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif

#if ENABLE_PICKLEDEVICE==1
#include "pickle_device_manager.h"
#endif


/* ================================================================
 *  Module-level state
 * ================================================================ */

#if ENABLE_PICKLEDEVICE==1
static PickleDeviceManager* pdev           = nullptr;
static volatile uint64_t*   UCPage_kern1   = nullptr;   /* SpMV q = A·p   */
static volatile uint64_t*   UCPage_kern2   = nullptr;   /* SpMV r = A·z   */
static volatile uint64_t*   PerfPage       = nullptr;
#endif

static uint64_t g_use_pdev          = 0;
static uint64_t g_prefetch_distance = 0;
static int      g_prefetch_mode     = 0;   /* 0 unknown, 1 single, 2 bulk */
static uint64_t g_bulk_chunk_size   = 0;


/* ================================================================
 *  extern "C" functions callable from Fortran
 * ================================================================ */

extern "C" {

void map_m5_mem_hook() {
    map_m5_mem();
}

void m5_exit_hook() {
    m5_exit_addr(0);
}

void wait_till_pdev_available() {
    uint64_t failure_count = 0;
    if (pdev == NULL) {
        pdev = new PickleDeviceManager();
    }

    /* Get performance monitoring page */
    if (PerfPage == NULL) {
        PerfPage = (volatile uint64_t*) pdev->getPerfPagePtr();
        printf("PerfPage: 0x%lx\n", (unsigned long)PerfPage);
        assert(PerfPage != NULL);
    }
    while (true) {
        /* Read device capabilities */
        PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
        g_use_pdev = specs.availability;

        if (!(g_use_pdev == 0 || g_use_pdev == 1)) {
            failure_count++;
            printf("  . Use pdev: %lu\n", (unsigned long)g_use_pdev);
            m5_exit_addr(0);
        } else {
            printf("  . Use pdev: %lu\n", (unsigned long)g_use_pdev);
            if (failure_count == 0) {
                m5_exit_addr(0);
            }
            return;
        }
    }
}

/* ----------------------------------------------------------------
 *  pickle_cg_init  —  create the device manager and read specs
 *
 *  All four output parameters are passed by reference (Fortran
 *  default calling convention).
 * ---------------------------------------------------------------- */
void pickle_cg_init(
    uint64_t* out_use_pdev,
    uint64_t* out_prefetch_distance,
    int*      out_prefetch_mode,
    uint64_t* out_bulk_chunk_size)
{
#if ENABLE_PICKLEDEVICE==1
    if (pdev == nullptr) {
        pdev = new PickleDeviceManager();
    }

    if (PerfPage == nullptr) {
        PerfPage = (volatile uint64_t*) pdev->getPerfPagePtr();
        printf("[Pickle CG] PerfPage : 0x%lx\n", (unsigned long)PerfPage);
        assert(PerfPage != nullptr);
    }

    PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
    g_use_pdev          = specs.availability;
    g_prefetch_distance = specs.prefetch_distance;
    g_prefetch_mode     = static_cast<int>(specs.prefetch_mode);
    g_bulk_chunk_size   = specs.bulk_mode_chunk_size;
#endif

    *out_use_pdev          = g_use_pdev;
    *out_prefetch_distance = g_prefetch_distance;
    *out_prefetch_mode     = g_prefetch_mode;
    *out_bulk_chunk_size   = g_bulk_chunk_size;

    printf("[Pickle CG] Device specs:\n");
    printf("  . use_pdev          : %lu\n", (unsigned long)g_use_pdev);
    printf("  . prefetch_distance : %lu\n", (unsigned long)g_prefetch_distance);
    printf("  . prefetch_mode     : %d   (0=unknown, 1=single, 2=bulk)\n",
           g_prefetch_mode);
    printf("  . bulk_chunk_size   : %lu\n", (unsigned long)g_bulk_chunk_size);
}


/* ----------------------------------------------------------------
 *  pickle_cg_setup_spmv  —  register one SpMV kernel with Pickle
 *
 *  kernel_id = 1  →  q = A·p  (inner CG iteration)
 *  kernel_id = 2  →  r = A·z  (residual computation)
 *
 *  Fortran passes all arguments by reference.
 * ---------------------------------------------------------------- */
void pickle_cg_setup_spmv(
    const int*     kernel_id,
    const void*    rowstr_base,       /* &rowstr(1)                */
    const int64_t* rowstr_nelems,     /* na + 1                    */
    const int64_t* rowstr_esz,        /* bytes per rowstr element  */
    const void*    colidx_base,       /* &colidx(1)                */
    const int64_t* colidx_nelems,     /* nz                        */
    const int64_t* colidx_esz,        /* bytes per colidx element  */
    const void*    vec_base,          /* &p(1) or &z(1)            */
    const int64_t* vec_nelems,        /* na + 2                    */
    const int64_t* vec_esz)           /* 8 (double precision)      */
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    char kname[64];
    snprintf(kname, sizeof(kname), "cg_spmv_kernel_%d", *kernel_id);

    PickleJob job(kname);

    /* ----------------------------------------------------------
     * rowstr  —  CSR row pointers  (Ranged / Pointer)
     *
     * No 1-based adjustment needed here: the hint value (j − 1)
     * is already 0-based, so  vaddr_start + hint × esz  gives
     * the address of rowstr(j).
     * ---------------------------------------------------------- */
    auto rowstr_desc = std::make_shared<PickleArrayDescriptor>();
    rowstr_desc->name         = "rowstr";
    rowstr_desc->vaddr_start  = (uint64_t)rowstr_base;
    rowstr_desc->vaddr_end    = (uint64_t)rowstr_base
                                + (*rowstr_nelems) * (*rowstr_esz);
    rowstr_desc->element_size = (uint64_t)(*rowstr_esz);
    rowstr_desc->access_type      = AccessType::Ranged;
    rowstr_desc->addressing_mode  = AddressingMode::Pointer;
    job.addArrayDescriptor(rowstr_desc);

    /* ----------------------------------------------------------
     * colidx  —  CSR column indices  (SingleElement / Index)
     *
     * Fortran 1-based adjustment: values stored in rowstr are
     * 1-based positions into colidx.  Shift vaddr_start back by
     * one element so the device's  base + V × esz  formula lands
     * on the correct address for 1-based V.
     * ---------------------------------------------------------- */
    auto colidx_desc = std::make_shared<PickleArrayDescriptor>();
    colidx_desc->name         = "colidx";
    colidx_desc->vaddr_start  = (uint64_t)colidx_base - (uint64_t)(*colidx_esz);
    colidx_desc->vaddr_end    = (uint64_t)colidx_base
                                + (*colidx_nelems) * (*colidx_esz);
    colidx_desc->element_size = (uint64_t)(*colidx_esz);
    colidx_desc->access_type      = AccessType::SingleElement;
    colidx_desc->addressing_mode  = AddressingMode::Index;
    job.addArrayDescriptor(colidx_desc);

    /* ----------------------------------------------------------
     * target vector (p or z)  —  dense vector  (SingleElement / Index)
     *
     * Fortran 1-based adjustment: values in colidx are 1-based
     * column indices into the target vector.
     * ---------------------------------------------------------- */
    auto vec_desc = std::make_shared<PickleArrayDescriptor>();
    vec_desc->name         = (*kernel_id == 1) ? "p_vec" : "z_vec";
    vec_desc->vaddr_start  = (uint64_t)vec_base - (uint64_t)(*vec_esz);
    vec_desc->vaddr_end    = (uint64_t)vec_base
                             + (*vec_nelems) * (*vec_esz);
    vec_desc->element_size = (uint64_t)(*vec_esz);
    vec_desc->access_type      = AccessType::SingleElement;
    vec_desc->addressing_mode  = AddressingMode::Index;
    job.addArrayDescriptor(vec_desc);

    /* Link the indirection chain:
     *   colidx values  ──index-into──►  target vector        */
    colidx_desc->dst_indexing_array_id = vec_desc->getArrayId();

    job.print();
    pdev->sendJob(job);
    printf("[Pickle CG] Sent %s\n", kname);
#else
    (void)kernel_id; (void)rowstr_base; (void)rowstr_nelems;
    (void)rowstr_esz; (void)colidx_base; (void)colidx_nelems;
    (void)colidx_esz; (void)vec_base; (void)vec_nelems; (void)vec_esz;
#endif
}


/* ----------------------------------------------------------------
 *  pickle_cg_setup_ucpages  —  obtain the UCPage communication area
 * ---------------------------------------------------------------- */
void pickle_cg_setup_ucpages(void)
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    uint64_t* base = (uint64_t*) pdev->getUCPagePtr(0);
    assert(base != nullptr);

    UCPage_kern1 = (volatile uint64_t*)(base);       /* kernel 1: q=A·p */
    UCPage_kern2 = (volatile uint64_t*)(base + 1);   /* kernel 2: r=A·z */

    printf("[Pickle CG] UCPage_kern1 : 0x%lx\n", (unsigned long)UCPage_kern1);
    printf("[Pickle CG] UCPage_kern2 : 0x%lx\n", (unsigned long)UCPage_kern2);
#endif
}


/* ----------------------------------------------------------------
 *  pickle_cg_get_ucpage_ptrs  —  export raw UCPage addresses
 *
 *  Returns the addresses of UCPage_kern1 and UCPage_kern2 so that
 *  Fortran can store hints directly via volatile pointers, avoiding
 *  per-row function call overhead.  This matches the graph benchmarks'
 *  inline  *UCPage = value  pattern.
 *
 *  Fortran side converts these via c_f_pointer() to volatile pointers.
 * ---------------------------------------------------------------- */
void pickle_cg_get_ucpage_ptrs(
    int64_t** out_kern1,
    int64_t** out_kern2)
{
#if ENABLE_PICKLEDEVICE==1
    *out_kern1 = (int64_t*)UCPage_kern1;
    *out_kern2 = (int64_t*)UCPage_kern2;
#else
    *out_kern1 = nullptr;
    *out_kern2 = nullptr;
#endif
}
 
 
/* ----------------------------------------------------------------
 *  pickle_cg_spmv_hint  —  send a row-level prefetch hint
 *
 *  DEPRECATED: prefer direct stores via pkl_ucpage_kern1/kern2
 *  volatile Fortran pointers obtained from pickle_cg_get_ucpage_ptrs.
 *  Kept for backward compatibility.
 *
 *  kernel_id : 1 (q=A·p) or 2 (r=A·z)
 *  row_0based: the 0-based row index  (Fortran j − 1)
 * ---------------------------------------------------------------- */
void pickle_cg_spmv_hint(const int* kernel_id, const int64_t* row_0based)
{
#if ENABLE_PICKLEDEVICE==1
    if (*kernel_id == 1) {
        *UCPage_kern1 = (uint64_t)(*row_0based);
    } else {
        *UCPage_kern2 = (uint64_t)(*row_0based);
    }
#else
    (void)kernel_id; (void)row_0based;
#endif
}

/* ----------------------------------------------------------------
 *  Performance monitoring  —  mirror the graph benchmarks
 * ---------------------------------------------------------------- */
void pickle_cg_perf_start(const int* thread_id)
{
#if ENABLE_PICKLEDEVICE==1
    if (PerfPage != nullptr)
        *PerfPage = ((uint64_t)(*thread_id) << 1) | 0u;
#else
    (void)thread_id;
#endif
}

void pickle_cg_perf_complete(const int* thread_id)
{
#if ENABLE_PICKLEDEVICE==1
    if (PerfPage != nullptr)
        *PerfPage = ((uint64_t)(*thread_id) << 1) | 1u;
#else
    (void)thread_id;
#endif
}


/* ----------------------------------------------------------------
 *  pickle_cg_finalize  —  tear down the device manager
 * ---------------------------------------------------------------- */
void pickle_cg_finalize(void)
{
#if ENABLE_PICKLEDEVICE==1
    if (pdev != nullptr) {
        delete pdev;
        pdev = nullptr;
    }
    UCPage_kern1 = nullptr;
    UCPage_kern2 = nullptr;
    PerfPage     = nullptr;
#endif
}


}  /* extern "C" */
