/*
 * pickle_ua_hooks.cpp
 *
 * C++ hooks for Pickle Prefetcher integration with NPB UA benchmark.
 * These functions have C linkage so they can be called from Fortran 90
 * via iso_c_binding.
 *
 * UA hot indirect-access pattern (transfer.f90, transf/transfb):
 *
 *   il = idel(i,j,iface,ie)             ! sequential read of idel
 *   ig = idmo(i,j,je1,je2,iface,ie)     ! sequential read of idmo
 *   tx(il)   = ... tmor(ig) ...         ! transf
 *   tmor(ig) = ... tx(il)  ...          ! transfb
 *
 *   idel/idmo  ≡  out_neighbors  (sequential, integer indices)
 *   tx / tmor  ≡  node property  (random target, double precision)
 *
 * Four kernels are registered for the diffusion CG inner loop, which
 * dominates the timed phase:
 *
 *   Kernel 1 : idel → pdiff   (transf  scatter target)
 *   Kernel 2 : idmo → pmorx   (transf  gather  source)
 *   Kernel 3 : idel → pdiffp  (transfb gather  source)
 *   Kernel 4 : idmo → ppmor   (transfb scatter target)
 *   Kernel 5 : num elements update
 *
 * Fortran 1-based indexing
 * ========================
 * The Pickle device assumes 0-based address arithmetic internally:
 *
 *     address = vaddr_start + index × element_size
 *
 * Fortran stores arr(V) at  &arr(1) + (V-1) × esz.  To make the
 * device's 0-based formula produce the correct address when it receives
 * a 1-based index V from idel or idmo, we shift the target vector's
 * vaddr_start back by one element:
 *
 *     vaddr_start  =  &arr(1) − esz          ("virtual element 0")
 *
 * The source arrays (idel/idmo) are read by the device using the hint
 * value as a 0-based index, and Fortran sends already-converted 0-based
 * offsets in the hot loop.
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
static volatile uint64_t*   UCPage_kern1   = nullptr;  /* idel → pdiff   */
static volatile uint64_t*   UCPage_kern2   = nullptr;  /* idmo → pmorx   */
static volatile uint64_t*   UCPage_kern3   = nullptr;  /* idel → pdiffp  */
static volatile uint64_t*   UCPage_kern4   = nullptr;  /* idmo → ppmor   */
static volatile uint64_t*   UCPage_kern5   = nullptr;  /* num elements update*/
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
#if ENABLE_GEM5==1
    map_m5_mem();
#endif
}

void m5_exit_hook() {
#if ENABLE_GEM5==1
    m5_exit_addr(0);
#endif
}

/* ----------------------------------------------------------------
 *  pickle_ua_init  —  create the device manager and read specs
 * ---------------------------------------------------------------- */
void pickle_ua_init(
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
        printf("[Pickle UA] PerfPage : 0x%lx\n", (unsigned long)PerfPage);
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

    printf("[Pickle UA] Device specs:\n");
    printf("  . use_pdev          : %lu\n", (unsigned long)g_use_pdev);
    printf("  . prefetch_distance : %lu\n", (unsigned long)g_prefetch_distance);
    printf("  . prefetch_mode     : %d   (0=unknown, 1=single, 2=bulk)\n",
           g_prefetch_mode);
    printf("  . bulk_chunk_size   : %lu\n", (unsigned long)g_bulk_chunk_size);
}


#if ENABLE_PICKLEDEVICE==1
/* Helper: register one dummy kernel. */
static void register_dummy_kernel(const char* kname) {
    PickleJob job(kname);
    auto src_desc = std::make_shared<PickleArrayDescriptor>();
    src_desc->name         = "dummy";
    src_desc->vaddr_start  = (uint64_t)0;
    src_desc->vaddr_end    = (uint64_t)0;
    src_desc->element_size = (uint64_t)0;
    src_desc->access_type      = AccessType::SingleElement;
    src_desc->addressing_mode  = AddressingMode::Index;
    job.addArrayDescriptor(src_desc);
    job.print();
    pdev->sendJob(job);
    printf("[Pickle UA] Sent %s\n", kname);
}
#endif


/* ----------------------------------------------------------------
 *  pickle_ua_setup_idel_kernel  —  register an (idel → tx) kernel
 *
 *  kernel_id = 1  →  idel → pdiff   (transf  scatter target)
 *  kernel_id = 3  →  idel → pdiffp  (transfb gather  source)
 * ---------------------------------------------------------------- */
void pickle_ua_setup_idel_kernel(
    const int*     kernel_id,
    const void*    idel_base,
    const int64_t* idel_nelems,
    const int64_t* idel_esz,
    const void*    tx_base,
    const int64_t* tx_nelems,
    const int64_t* tx_esz)
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    char kname[64];
    snprintf(kname, sizeof(kname), "ua_idel_kernel_%d", *kernel_id);
    const char* tgt_name = (*kernel_id == 1) ? "pdiff" : "pdiffp";
    PickleJob job(kname);

    auto src_desc = std::make_shared<PickleArrayDescriptor>();
    src_desc->name = "idel";
    src_desc->vaddr_start = (uint64_t)idel_base;
    src_desc->vaddr_end = (uint64_t)idel_base + *idel_nelems * *idel_esz;
    src_desc->element_size = (uint64_t)*idel_esz;
    src_desc->access_type = AccessType::SingleElement;
    src_desc->addressing_mode = AddressingMode::Index;
    job.addArrayDescriptor(src_desc);

    auto tgt_desc = std::make_shared<PickleArrayDescriptor>();
    tgt_desc->name = tgt_name; // tmor
    tgt_desc->vaddr_start = (uint64_t)tx_base;
    tgt_desc->vaddr_end = (uint64_t)tx_base + *tx_nelems * *tx_esz;
    tgt_desc->element_size = (uint64_t)*tx_esz;
    tgt_desc->access_type = AccessType::SingleElement;
    tgt_desc->addressing_mode = AddressingMode::Index;
    job.addArrayDescriptor(tgt_desc);

    /* Indirection chain: src values index into target vector. */
    src_desc->dst_indexing_array_id = tgt_desc->getArrayId();

    job.print();
    pdev->sendJob(job);
    printf("[Pickle UA] Sent %s\n", kname);

#endif
}


/* ----------------------------------------------------------------
 *  pickle_ua_setup_idmo_kernel  —  register an (idmo → tmor) kernel
 *
 *  kernel_id = 2  →  idmo → pmorx   (transf  gather  source)
 *  kernel_id = 4  →  idmo → ppmor   (transfb scatter target)
 * ---------------------------------------------------------------- */
void pickle_ua_setup_idmo_kernel(
    const int*     kernel_id,
    const void*    idmo_base,
    const int64_t* idmo_nelems,
    const int64_t* idmo_esz,
    const void*    tmor_base,
    const int64_t* tmor_nelems,
    const int64_t* tmor_esz,
    const void*    cbc_base,
    const int64_t* cbc_nelems,
    const int64_t* cbc_esz
)
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    char kname[64];
    snprintf(kname, sizeof(kname), "ua_idmo_kernel_%d", *kernel_id);
    const char* tgt_name = (*kernel_id == 2) ? "pmorx" : "ppmor";

    PickleJob job(kname);

    auto src_desc = std::make_shared<PickleArrayDescriptor>();
    src_desc->name = "idmo";
    src_desc->vaddr_start = (uint64_t)idmo_base;
    src_desc->vaddr_end = (uint64_t)idmo_base + *idmo_nelems * *idmo_esz;
    src_desc->element_size = (uint64_t)*idmo_esz;
    src_desc->access_type = AccessType::SingleElement;
    src_desc->addressing_mode = AddressingMode::Index;
    job.addArrayDescriptor(src_desc);

    auto tgt_desc = std::make_shared<PickleArrayDescriptor>();
    tgt_desc->name = tgt_name; // tmor
    tgt_desc->vaddr_start = (uint64_t)tmor_base;
    tgt_desc->vaddr_end = (uint64_t)tmor_base + *tmor_nelems * *tmor_esz;
    tgt_desc->element_size = (uint64_t)*tmor_esz;
    tgt_desc->access_type = AccessType::SingleElement;
    tgt_desc->addressing_mode = AddressingMode::Index;
    job.addArrayDescriptor(tgt_desc);

    /* Indirection chain: src values index into target vector. */
    src_desc->dst_indexing_array_id = tgt_desc->getArrayId();

    // We send the cbc array for branch decision
    auto cond_desc = std::make_shared<PickleArrayDescriptor>();
    cond_desc->name = "cbc";
    cond_desc->vaddr_start = (uint64_t)cbc_base;
    cond_desc->vaddr_end = (uint64_t)cbc_base + *cbc_nelems * *cbc_esz;
    cond_desc->element_size = (uint64_t)*cbc_esz;
    cond_desc->access_type = AccessType::SingleElement;
    cond_desc->addressing_mode = AddressingMode::Index;
    job.addArrayDescriptor(cond_desc);

    job.print();
    pdev->sendJob(job);
    printf("[Pickle UA] Sent %s\n", kname);

    //register_index_to_vec_kernel(
    //    kname,
    //    "idmo", idmo_base, *idmo_nelems, *idmo_esz,
    //    tgt_name, tmor_base, *tmor_nelems, *tmor_esz);
#else
    (void)kernel_id; (void)idmo_base; (void)idmo_nelems; (void)idmo_esz;
    (void)tmor_base; (void)tmor_nelems; (void)tmor_esz;
#endif
}

void pickle_ua_num_elements_update_kernel()
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    register_dummy_kernel("ua_num_elements_update_kernel");
#endif
}


/* ----------------------------------------------------------------
 *  pickle_ua_setup_ucpages  —  obtain the UCPage communication area
 *
 *  Slots 0..3 carry hints for kernels 1..4 respectively.
 * ---------------------------------------------------------------- */
void pickle_ua_setup_ucpages(void)
{
#if ENABLE_PICKLEDEVICE==1
    if (g_use_pdev != 1) return;

    uint64_t* base = (uint64_t*) pdev->getUCPagePtr(0);
    assert(base != nullptr);

    UCPage_kern1 = (volatile uint64_t*)(base);
    UCPage_kern2 = (volatile uint64_t*)(base + 1);
    UCPage_kern3 = (volatile uint64_t*)(base + 2);
    UCPage_kern4 = (volatile uint64_t*)(base + 3);
    UCPage_kern5 = (volatile uint64_t*)(base + 4);

    printf("[Pickle UA] UCPage_kern1 : 0x%lx\n", (unsigned long)UCPage_kern1);
    printf("[Pickle UA] UCPage_kern2 : 0x%lx\n", (unsigned long)UCPage_kern2);
    printf("[Pickle UA] UCPage_kern3 : 0x%lx\n", (unsigned long)UCPage_kern3);
    printf("[Pickle UA] UCPage_kern4 : 0x%lx\n", (unsigned long)UCPage_kern4);
    printf("[Pickle UA] UCPage_kern5 : 0x%lx\n", (unsigned long)UCPage_kern5);
#endif
}


/* ----------------------------------------------------------------
 *  pickle_ua_get_ucpage_ptrs  —  export raw UCPage addresses so
 *  Fortran can store hints directly via volatile pointers.
 * ---------------------------------------------------------------- */
void pickle_ua_get_ucpage_ptrs(
    int64_t** out_kern1,
    int64_t** out_kern2,
    int64_t** out_kern3,
    int64_t** out_kern4,
    int64_t** out_kern5)
{
#if ENABLE_PICKLEDEVICE==1
    *out_kern1 = (int64_t*)UCPage_kern1;
    *out_kern2 = (int64_t*)UCPage_kern2;
    *out_kern3 = (int64_t*)UCPage_kern3;
    *out_kern4 = (int64_t*)UCPage_kern4;
    *out_kern5 = (int64_t*)UCPage_kern5;
#else
    *out_kern1 = nullptr;
    *out_kern2 = nullptr;
    *out_kern3 = nullptr;
    *out_kern4 = nullptr;
    *out_kern5 = nullptr;
#endif
}


/* ----------------------------------------------------------------
 *  Performance monitoring  —  mirror the graph benchmarks
 * ---------------------------------------------------------------- */
void pickle_ua_perf_start(const int* thread_id)
{
#if ENABLE_PICKLEDEVICE==1
    if (PerfPage != nullptr)
        *PerfPage = ((uint64_t)(*thread_id) << 1) | 0u;
#else
    (void)thread_id;
#endif
}

void pickle_ua_perf_complete(const int* thread_id)
{
#if ENABLE_PICKLEDEVICE==1
    if (PerfPage != nullptr)
        *PerfPage = ((uint64_t)(*thread_id) << 1) | 1u;
#else
    (void)thread_id;
#endif
}


/* ----------------------------------------------------------------
 *  pickle_ua_finalize  —  tear down the device manager
 * ---------------------------------------------------------------- */
void pickle_ua_finalize(void)
{
#if ENABLE_PICKLEDEVICE==1
    if (pdev != nullptr) {
        delete pdev;
        pdev = nullptr;
    }
    UCPage_kern1 = nullptr;
    UCPage_kern2 = nullptr;
    UCPage_kern3 = nullptr;
    UCPage_kern4 = nullptr;
    UCPage_kern5 = nullptr;
    PerfPage     = nullptr;
#endif
}


}  /* extern "C" */
