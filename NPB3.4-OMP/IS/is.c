/*************************************************************************
 *                                                                       *
 *       N  A  S     P A R A L L E L     B E N C H M A R K S  3.4        *
 *                                                                       *
 *                      O p e n M P     V E R S I O N                    *
 *                                                                       *
 *                                  I S                                  *
 *                                                                       *
 *************************************************************************
 *                                                                       *
 *   This benchmark is an OpenMP version of the NPB IS code.             *
 *   It is described in NAS Technical Report 99-011.                     *
 *                                                                       *
 *   Permission to use, copy, distribute and modify this software        *
 *   for any purpose with or without fee is hereby granted.  We          *
 *   request, however, that all derived work reference the NAS           *
 *   Parallel Benchmarks 3.4. This software is provided "as is"         *
 *   without express or implied warranty.                                *
 *                                                                       *
 *   Information on NPB 3.4, including the technical report, the         *
 *   original specifications, source code, results and information       *
 *   on how to submit new results, is available at:                      *
 *                                                                       *
 *          http://www.nas.nasa.gov/Software/NPB/                        *
 *                                                                       *
 *   Send comments or suggestions to  npb@nas.nasa.gov                   *
 *                                                                       *
 *         NAS Parallel Benchmarks Group                                 *
 *         NASA Ames Research Center                                     *
 *         Mail Stop: T27A-1                                             *
 *         Moffett Field, CA   94035-1000                                *
 *                                                                       *
 *         E-mail:  npb@nas.nasa.gov                                     *
 *         Fax:     (650) 604-3957                                       *
 *                                                                       *
 *************************************************************************
 *                                                                       *
 *   Author: M. Yarrow                                                   *
 *           H. Jin                                                      *
 *                                                                       *
 *************************************************************************/

/*
 * Pickle Prefetcher Integration Notes
 * ====================================
 *
 * The IS (Integer Sort) benchmark performs counting sort using bucket
 * decomposition. The key memory access pattern that benefits from
 * Pickle prefetching is the per-bucket ranking loop:
 *
 *     for (k = m; k < bucket_ptrs[i]; k++)
 *         key_buff_ptr[key_buff_ptr2[k]]++;
 *
 * This is a classic indirect/gather pattern:
 *   - key_buff_ptr2 (= key_buff2) is iterated SEQUENTIALLY
 *   - Each value key_buff_ptr2[k] is used as an INDEX into key_buff_ptr
 *     (= key_buff1), causing random accesses
 *
 * This is analogous to graph traversal patterns where sequential
 * iteration through a neighbor list drives random property lookups:
 *   - key_buff_ptr2  ~  out_neighbors array (sequential source)
 *   - key_buff_ptr   ~  node property array (indirect target)
 *
 * Array descriptor chain for Pickle:
 *   key_buff_ptr2 [SingleElement, Index]
 *       └── dst_indexing_array_id ──► key_buff_ptr [SingleElement, Index]
 *
 * The prefetcher reads ahead in key_buff_ptr2 by prefetch_distance,
 * extracts the key value, and prefetches the corresponding cache line
 * in key_buff_ptr before the core needs it.
 */

#include "npbparams.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_device_manager.h"
#endif

#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif /* ENABLE_GEM5 */


/*****************************************************************/
/* Bucket sort configuration                                     */
/*****************************************************************/
#define USE_BUCKETS

/* Uncomment below for cyclic schedule */
/*#define SCHED_CYCLIC*/

/******************/
/* default values */
/******************/
#ifndef CLASS
#define CLASS 'S'
#endif


/*************/
/*  CLASS S  */
/*************/
#if CLASS == 'S'
#define  TOTAL_KEYS_LOG_2    16
#define  MAX_KEY_LOG_2       11
#define  NUM_BUCKETS_LOG_2   9
#endif


/*************/
/*  CLASS W  */
/*************/
#if CLASS == 'W'
#define  TOTAL_KEYS_LOG_2    20
#define  MAX_KEY_LOG_2       16
#define  NUM_BUCKETS_LOG_2   10
#endif

/*************/
/*  CLASS A  */
/*************/
#if CLASS == 'A'
#define  TOTAL_KEYS_LOG_2    23
#define  MAX_KEY_LOG_2       19
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS B  */
/*************/
#if CLASS == 'B'
#define  TOTAL_KEYS_LOG_2    25
#define  MAX_KEY_LOG_2       21
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS C  */
/*************/
#if CLASS == 'C'
#define  TOTAL_KEYS_LOG_2    27
#define  MAX_KEY_LOG_2       23
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS D  */
/*************/
#if CLASS == 'D'
#define  TOTAL_KEYS_LOG_2    31
#define  MAX_KEY_LOG_2       27
#define  NUM_BUCKETS_LOG_2   10
#endif


/*************/
/*  CLASS E  */
/*************/
#if CLASS == 'E'
#define  TOTAL_KEYS_LOG_2    35
#define  MAX_KEY_LOG_2       31
#define  NUM_BUCKETS_LOG_2   10
#endif


#if (CLASS == 'D' || CLASS == 'E')
#define  TOTAL_KEYS          (1L << TOTAL_KEYS_LOG_2)
#define  TOTAL_KS1           (1 << (TOTAL_KEYS_LOG_2-8))
#define  TOTAL_KS2           (1 << 8)
#define  MAX_KEY             (1L << MAX_KEY_LOG_2)
#else
#define  TOTAL_KEYS          (1 << TOTAL_KEYS_LOG_2)
#define  TOTAL_KS1           TOTAL_KEYS
#define  TOTAL_KS2           1
#define  MAX_KEY             (1 << MAX_KEY_LOG_2)
#endif
#define  NUM_BUCKETS         (1 << NUM_BUCKETS_LOG_2)
#define  NUM_KEYS            TOTAL_KEYS
#define  SIZE_OF_BUFFERS     NUM_KEYS


#define  MAX_ITERATIONS      10
#define  TEST_ARRAY_SIZE     5


/*************************************/
/* Typedef: if necessary, change the */
/* size of int here by changing the  */
/* int type to, say, long            */
/*************************************/
#if (CLASS == 'D' || CLASS == 'E')
typedef  long INT_TYPE;
#else
typedef  int  INT_TYPE;
#endif


/********************/
/* Some global info */
/********************/
INT_TYPE *key_buff_ptr_global;         /* used by full_verify to get */
                                       /* copies of rank info        */

int      passed_verification;


/************************************/
/* These are the three main arrays. */
/* See SIZE_OF_BUFFERS def above    */
/************************************/
INT_TYPE key_array[SIZE_OF_BUFFERS],
         key_buff1[MAX_KEY],
         key_buff2[SIZE_OF_BUFFERS],
         partial_verify_vals[TEST_ARRAY_SIZE],
         **key_buff1_aptr = NULL;

#ifdef USE_BUCKETS
INT_TYPE **bucket_size,
         bucket_ptrs[NUM_BUCKETS];
#pragma omp threadprivate(bucket_ptrs)
#endif


/************************************/
/* Pickle Prefetcher globals        */
/************************************/
#if ENABLE_PICKLEDEVICE==1
volatile uint64_t* UCPage = NULL;
volatile uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

/* Pickle device manager - global instance */
PickleDeviceManager* pdev = NULL;

/* Device specs - populated during setup */
uint64_t use_pdev = 0;
uint64_t prefetch_distance = 0;
PrefetchMode prefetch_mode = UNKNOWN;
uint64_t bulk_mode_chunk_size = 0;
#endif


/**********************/
/* Partial verif info */
/**********************/
INT_TYPE test_index_array[TEST_ARRAY_SIZE],
         test_rank_array[TEST_ARRAY_SIZE];

int      S_test_index_array[TEST_ARRAY_SIZE] =
                             {48427,17148,23627,62548,4431},
         S_test_rank_array[TEST_ARRAY_SIZE] =
                             {0,18,346,64917,65463},

         W_test_index_array[TEST_ARRAY_SIZE] =
                             {357773,934767,875723,898999,404505},
         W_test_rank_array[TEST_ARRAY_SIZE] =
                             {1249,11698,1039987,1043896,1048018},

         A_test_index_array[TEST_ARRAY_SIZE] =
                             {2112377,662041,5336171,3642833,4250760},
         A_test_rank_array[TEST_ARRAY_SIZE] =
                             {104,17523,123928,8288932,8388264},

         B_test_index_array[TEST_ARRAY_SIZE] =
                             {41869,812306,5102857,18232239,26860214},
         B_test_rank_array[TEST_ARRAY_SIZE] =
                             {33422937,10244,59149,33135281,99},

         C_test_index_array[TEST_ARRAY_SIZE] =
                             {44172927,72999161,74326391,129606274,21736814},
         C_test_rank_array[TEST_ARRAY_SIZE] =
                             {61147,882988,266290,133997595,133525895};

long     D_test_index_array[TEST_ARRAY_SIZE] =
                             {1317351170,995930646,1157283250,1503301535,1453734525},
         D_test_rank_array[TEST_ARRAY_SIZE] =
                             {1,36538729,1978098519,2145192618,2147425337},

         E_test_index_array[TEST_ARRAY_SIZE] =
                             {21492309536L,24606226181L,12608530949L,4065943607L,3324513396L},
         E_test_rank_array[TEST_ARRAY_SIZE] =
                             {3L,27580354L,3248475153L,30048754302L,31485259697L};


/***********************/
/* function prototypes */
/***********************/
double	randlc( double *X, double *A );

void full_verify( void );

void c_print_results( char   *name,
                      char   workload_class,
                      int    n1,
                      int    n2,
                      int    n3,
                      int    niter,
                      double t,
                      double mops,
		      char   *optype,
                      int    passed_verification,
                      char   *npbversion,
                      char   *compiletime,
                      char   *cc,
                      char   *clink,
                      char   *c_lib,
                      char   *c_inc,
                      char   *cflags,
                      char   *clinkflags );

#include "../common/c_timers.h"


/*****************************************************************/
/*************           R  A  N  D  L  C             ************/
/*****************************************************************/

static int      KS=0;
static double	R23, R46, T23, T46;
#pragma omp threadprivate(KS, R23, R46, T23, T46)

double	randlc( double *X, double *A )
{
      double		T1, T2, T3, T4;
      double		A1;
      double		A2;
      double		X1;
      double		X2;
      double		Z;
      int     		i, j;

      if (KS == 0)
      {
        R23 = 1.0;
        R46 = 1.0;
        T23 = 1.0;
        T46 = 1.0;

        for (i=1; i<=23; i++)
        {
          R23 = 0.50 * R23;
          T23 = 2.0 * T23;
        }
        for (i=1; i<=46; i++)
        {
          R46 = 0.50 * R46;
          T46 = 2.0 * T46;
        }
        KS = 1;
      }

      T1 = R23 * *A;
      j  = T1;
      A1 = j;
      A2 = *A - T23 * A1;

      T1 = R23 * *X;
      j  = T1;
      X1 = j;
      X2 = *X - T23 * X1;
      T1 = A1 * X2 + A2 * X1;

      j  = R23 * T1;
      T2 = j;
      Z = T1 - T23 * T2;
      T3 = T23 * Z + A2 * X2;
      j  = R46 * T3;
      T4 = j;
      *X = T3 - T46 * T4;
      return(R46 * *X);
}


/*****************************************************************/
/************   F  I  N  D  _  M  Y  _  S  E  E  D    ************/
/*****************************************************************/

double   find_my_seed( int kn,
                       int np,
                       long nn,
                       double s,
                       double a )
{
      double t1,t2;
      long   mq,nq,kk,ik;

      if ( kn == 0 ) return s;

      mq = (nn/4 + np - 1) / np;
      nq = mq * 4 * kn;

      t1 = s;
      t2 = a;
      kk = nq;
      while ( kk > 1 ) {
      	 ik = kk / 2;
         if( 2 * ik ==  kk ) {
            (void)randlc( &t2, &t2 );
	    kk = ik;
	 }
	 else {
            (void)randlc( &t1, &t2 );
	    kk = kk - 1;
	 }
      }
      (void)randlc( &t1, &t2 );

      return( t1 );
}


/*****************************************************************/
/*************      C  R  E  A  T  E  _  S  E  Q      ************/
/*****************************************************************/

void	create_seq( double seed, double a )
{
	double x, s;
	INT_TYPE i, k;

#pragma omp parallel private(x,s,i,k)
    {
	INT_TYPE k1, k2;
	double an = a;
	int myid = 0, num_threads = 1;
        INT_TYPE mq;

#ifdef _OPENMP
	myid = omp_get_thread_num();
	num_threads = omp_get_num_threads();
#endif

	mq = (NUM_KEYS + num_threads - 1) / num_threads;
	k1 = mq * myid;
	k2 = k1 + mq;
	if ( k2 > NUM_KEYS ) k2 = NUM_KEYS;

	KS = 0;
	s = find_my_seed( myid, num_threads,
			  (long)4*NUM_KEYS, seed, an );

        k = MAX_KEY/4;

	for (i=k1; i<k2; i++)
	{
	    x = randlc(&s, &an);
	    x += randlc(&s, &an);
    	    x += randlc(&s, &an);
	    x += randlc(&s, &an);

            key_array[i] = k*x;
	}
    } /*omp parallel*/
}


/*****************************************************************/
/*****************    Allocate Working Buffer     ****************/
/*****************************************************************/
void *alloc_mem( size_t size )
{
    void *p;

    p = (void *)malloc(size);
    if (!p) {
        perror("Memory allocation error");
        exit(1);
    }
    return p;
}

void alloc_key_buff( void )
{
    INT_TYPE i;
    int      num_threads = 1;

#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

#ifdef USE_BUCKETS
    bucket_size = (INT_TYPE **)alloc_mem(sizeof(INT_TYPE *) * num_threads);

    for (i = 0; i < num_threads; i++) {
        bucket_size[i] = (INT_TYPE *)alloc_mem(sizeof(INT_TYPE) * NUM_BUCKETS);
    }

    #pragma omp parallel for
    for( i=0; i<NUM_KEYS; i++ )
        key_buff2[i] = 0;

#else /*USE_BUCKETS*/

    key_buff1_aptr = (INT_TYPE **)alloc_mem(sizeof(INT_TYPE *) * num_threads);

    key_buff1_aptr[0] = key_buff1;
    for (i = 1; i < num_threads; i++) {
        key_buff1_aptr[i] = (INT_TYPE *)alloc_mem(sizeof(INT_TYPE) * MAX_KEY);
    }

#endif /*USE_BUCKETS*/
}


/*****************************************************************/
/*************    F  U  L  L  _  V  E  R  I  F  Y     ************/
/*****************************************************************/

void full_verify( void )
{
    INT_TYPE   i, j;
    INT_TYPE   k, k1, k2;

#ifdef USE_BUCKETS

#ifdef SCHED_CYCLIC
    #pragma omp parallel for private(i,j,k,k1) schedule(static,1)
#else
    #pragma omp parallel for private(i,j,k,k1) schedule(dynamic)
#endif
    for( j=0; j< NUM_BUCKETS; j++ ) {

        k1 = (j > 0)? bucket_ptrs[j-1] : 0;
        for ( i = k1; i < bucket_ptrs[j]; i++ ) {
            k = --key_buff_ptr_global[key_buff2[i]];
            key_array[k] = key_buff2[i];
        }
    }

#else

#pragma omp parallel private(i,j,k,k1,k2)
  {
    #pragma omp for
    for( i=0; i<NUM_KEYS; i++ )
        key_buff2[i] = key_array[i];

#ifdef _OPENMP
    j = omp_get_num_threads();
    j = (MAX_KEY + j - 1) / j;
    k1 = j * omp_get_thread_num();
#else
    j = MAX_KEY;
    k1 = 0;
#endif
    k2 = k1 + j;
    if (k2 > MAX_KEY) k2 = MAX_KEY;

    for( i=0; i<NUM_KEYS; i++ ) {
        if (key_buff2[i] >= k1 && key_buff2[i] < k2) {
            k = --key_buff_ptr_global[key_buff2[i]];
            key_array[k] = key_buff2[i];
        }
    }
  } /*omp parallel*/

#endif

    j = 0;
    #pragma omp parallel for reduction(+:j)
    for( i=1; i<NUM_KEYS; i++ )
        if( key_array[i-1] > key_array[i] )
            j++;

    if( j != 0 )
        printf( "Full_verify: number of keys out of sort: %ld\n", (long)j );
    else
        passed_verification++;
}


#if ENABLE_PICKLEDEVICE==1
/*****************************************************************/
/*************    P I C K L E   S E T U P             ************/
/*****************************************************************/

/*
 * Setup the Pickle device for the IS ranking kernel.
 *
 * The ranking kernel has the following indirect access pattern:
 *
 *   key_buff_ptr[ key_buff_ptr2[k] ]++
 *   ~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~
 *    target arr    source arr (sequential iteration)
 *
 * Array chain:
 *   key_buff2 (source, sequential) ──values-as-indices──► key_buff1 (target, random)
 *
 * This is analogous to the graph pattern:
 *   property[ neighbors[k] ]
 *   ~~~~~~~~  ~~~~~~~~~~~~
 *   node prop  neighbor list
 */
void wait_till_pickle_device_available( void )
{
#pragma message("abcd")
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
        use_pdev = specs.availability;

        if (!(use_pdev == 0 || use_pdev == 1)) {
            failure_count++;
            printf("  . Use pdev: %lu\n", (unsigned long)use_pdev);
            m5_exit_addr(0);
        } else {
            printf("  . Use pdev: %lu\n", (unsigned long)use_pdev);
            if (failure_count == 0) {
                m5_exit_addr(0);
            }
            return;
        }
    }
}
void pickle_setup( void )
{
    if (pdev == NULL) {
        pdev = new PickleDeviceManager();
    }

    /* Get performance monitoring page */
    if (PerfPage == NULL) {
        PerfPage = (volatile uint64_t*) pdev->getPerfPagePtr();
        printf("PerfPage: 0x%lx\n", (unsigned long)PerfPage);
        assert(PerfPage != NULL);
    }

    /* Read device capabilities */
    PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
    use_pdev = specs.availability;
    prefetch_distance = specs.prefetch_distance;
    prefetch_mode = specs.prefetch_mode;
    bulk_mode_chunk_size = specs.bulk_mode_chunk_size;

    printf("Pickle Device specs:\n");
    printf("  . Use pdev: %lu\n", (unsigned long)use_pdev);
    printf("  . Prefetch distance: %lu\n", (unsigned long)prefetch_distance);
    printf("  . Prefetch mode (0: unknown, 1: single, 2: bulk): %d\n", prefetch_mode);
    printf("  . Chunk size (should be non-zero in bulk mode): %lu\n",
           (unsigned long)bulk_mode_chunk_size);

    if (use_pdev == 1) {
        /*
         * Create and send the Pickle job describing the IS ranking kernel.
         *
         * The kernel iterates sequentially through key_buff2 (the bucket-
         * scattered keys), and for each key value, accesses key_buff1[key]
         * to update a histogram count. Pickle can read ahead in key_buff2
         * to discover future key values and prefetch the corresponding
         * key_buff1 cache lines.
         */
        PickleJob job(/*kernel_name*/"is_ranking_kernel");

        /* Source array: key_buff2 (iterated sequentially within each bucket) */
        PickleArrayDescriptor* key_buff2_desc = new PickleArrayDescriptor();
        key_buff2_desc->name = "key_buff2";
        key_buff2_desc->vaddr_start = (uint64_t)(&key_buff2[0]);
        key_buff2_desc->vaddr_end = (uint64_t)(&key_buff2[SIZE_OF_BUFFERS]);
        key_buff2_desc->element_size = sizeof(INT_TYPE);
        key_buff2_desc->access_type = AccessType::SingleElement;
        key_buff2_desc->addressing_mode = AddressingMode::Index;

        /* Target array: key_buff1 (accessed indirectly via values in key_buff2) */
        PickleArrayDescriptor* key_buff1_desc = new PickleArrayDescriptor();
        key_buff1_desc->name = "key_buff1";
        key_buff1_desc->vaddr_start = (uint64_t)(&key_buff1[0]);
        key_buff1_desc->vaddr_end = (uint64_t)(&key_buff1[MAX_KEY]);
        key_buff1_desc->element_size = sizeof(INT_TYPE);
        key_buff1_desc->access_type = AccessType::SingleElement;
        key_buff1_desc->addressing_mode = AddressingMode::Index;

        /*
         * Link: values read from key_buff2 are used as indices into key_buff1.
         * This tells Pickle: "when you read ahead in key_buff2 and find value V,
         * prefetch key_buff1[V]."
         */
        key_buff2_desc->dst_indexing_array_id = key_buff1_desc->getArrayId();

        std::shared_ptr<PickleArrayDescriptor> key_buff2_shared(key_buff2_desc);
        std::shared_ptr<PickleArrayDescriptor> key_buff1_shared(key_buff1_desc);
        job.addArrayDescriptor(key_buff2_shared);
        job.addArrayDescriptor(key_buff1_shared);

        job.print();
        pdev->sendJob(job);
        printf("Sent is_ranking_kernel job\n");

        /* Get the uncacheable communication page */
        UCPage = (volatile uint64_t*) pdev->getUCPagePtr(0);
        printf("UCPage: 0x%lx\n", (unsigned long)UCPage);
        assert(UCPage != NULL);
    }
}

void pickle_teardown( void )
{
    if (pdev != NULL) {
        delete pdev;
        pdev = NULL;
    }
}
#endif /* ENABLE_PICKLEDEVICE */


/*****************************************************************/
/*************             R  A  N  K             ****************/
/*****************************************************************/

void rank( int iteration )
{

    INT_TYPE    i, k;
    INT_TYPE    *key_buff_ptr, *key_buff_ptr2;

#ifdef USE_BUCKETS
    int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
    INT_TYPE num_bucket_keys = (1L << shift);
#endif


    key_array[iteration] = iteration;
    key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;


/*  Determine where the partial verify test keys are, load into  */
/*  top of array bucket_size                                     */
    for( i=0; i<TEST_ARRAY_SIZE; i++ )
        partial_verify_vals[i] = key_array[test_index_array[i]];


/*  Setup pointers to key buffers  */
#ifdef USE_BUCKETS
    key_buff_ptr2 = key_buff2;
#else
    key_buff_ptr2 = key_array;
#endif
    key_buff_ptr = key_buff1;


#pragma omp parallel private(i, k)
  {
    INT_TYPE *work_buff, m, k1, k2;
    int myid = 0, num_threads = 1;

#ifdef _OPENMP
    myid = omp_get_thread_num();
    num_threads = omp_get_num_threads();
#endif

#if ENABLE_PICKLEDEVICE==1
    const uint64_t thread_id = (uint64_t)myid;
    if (PerfPage != NULL)
        *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif

#ifdef USE_BUCKETS

    work_buff = bucket_size[myid];

/*  Initialize */
    for( i=0; i<NUM_BUCKETS; i++ )
        work_buff[i] = 0;

/*  Determine the number of keys in each bucket */
    #pragma omp for schedule(static)
    for( i=0; i<NUM_KEYS; i++ )
        work_buff[key_array[i] >> shift]++;

/*  Accumulative bucket sizes are the bucket pointers.
    These are global sizes accumulated upon to each bucket */
    bucket_ptrs[0] = 0;
    for( k=0; k< myid; k++ )
        bucket_ptrs[0] += bucket_size[k][0];

    for( i=1; i< NUM_BUCKETS; i++ ) {
        bucket_ptrs[i] = bucket_ptrs[i-1];
        for( k=0; k< myid; k++ )
            bucket_ptrs[i] += bucket_size[k][i];
        for( k=myid; k< num_threads; k++ )
            bucket_ptrs[i] += bucket_size[k][i-1];
    }


/*  Sort into appropriate bucket */
    #pragma omp for schedule(static)
    for( i=0; i<NUM_KEYS; i++ )
    {
        k = key_array[i];
        key_buff2[bucket_ptrs[k >> shift]++] = k;
    }

/*  The bucket pointers now point to the final accumulated sizes */
    if (myid < num_threads-1) {
        for( i=0; i< NUM_BUCKETS; i++ )
            for( k=myid+1; k< num_threads; k++ )
                bucket_ptrs[i] += bucket_size[k][i];
    }


/*  Now, buckets are sorted.  We only need to sort keys inside
    each bucket, which can be done in parallel.  Because the distribution
    of the number of keys in the buckets is Gaussian, the use of
    a dynamic schedule should improve load balance, thus, performance     */

#ifdef SCHED_CYCLIC
    #pragma omp for schedule(static,1)
#else
    #pragma omp for schedule(dynamic, 16384)
#endif
    for( i=0; i< NUM_BUCKETS; i++ ) {

/*  Clear the work array section associated with each bucket */
        k1 = i * num_bucket_keys;
        k2 = k1 + num_bucket_keys;
        for ( k = k1; k < k2; k++ )
            key_buff_ptr[k] = 0;

/*  Ranking of all keys occurs in this section:                 */
/*  In this section, the keys themselves are used as their
    own indexes to determine how many of each there are: their
    individual population                                       */
        m = (i > 0)? bucket_ptrs[i-1] : 0;

        /*
         * === PICKLE PREFETCH HINT FOR RANKING LOOP ===
         *
         * This is the performance-critical indirect access:
         *   key_buff_ptr[ key_buff_ptr2[k] ]++
         *
         * key_buff_ptr2 is iterated sequentially (k goes from m to bucket_ptrs[i]).
         * Each value key_buff_ptr2[k] is a key in range [k1, k2) and is used
         * to index into key_buff_ptr, causing potential cache misses for large
         * key ranges.
         *
         * The Pickle prefetcher reads ahead in key_buff_ptr2 by prefetch_distance,
         * discovers the future key value, and prefetches the target cache line in
         * key_buff_ptr before the core needs it.
         */
#if ENABLE_PICKLEDEVICE==1
        if (use_pdev == 1) {
            const INT_TYPE next_ptr = bucket_ptrs[i];
            for ( k = m; k < next_ptr; k++ ) {
                /* Send prefetch hint: address of current source element.
                 * The device reads key_buff_ptr2[k + prefetch_distance],
                 * gets the key value V, and prefetches key_buff_ptr[V]. */
                if ( k + prefetch_distance < next_ptr ) {
                    *UCPage = (uint64_t)(k);
                }
                key_buff_ptr[key_buff_ptr2[k]]++;
            }
        } else
#endif
        {
            /* Original loop without prefetch hints */
            for ( k = m; k < bucket_ptrs[i]; k++ )
                key_buff_ptr[key_buff_ptr2[k]]++;
        }

/*  To obtain ranks of each key, successively add the individual key
    population, not forgetting to add m, the total of lesser keys,
    to the first key population                                          */
        key_buff_ptr[k1] += m;
        for ( k = k1+1; k < k2; k++ )
            key_buff_ptr[k] += key_buff_ptr[k-1];

    }

#else /*USE_BUCKETS*/

    work_buff = key_buff1_aptr[myid];

/*  Clear the work array */
    for( i=0; i<MAX_KEY; i++ )
        work_buff[i] = 0;

/*  Ranking of all keys occurs in this section:                 */
/*  In this section, the keys themselves are used as their
    own indexes to determine how many of each there are: their
    individual population                                       */

    /*
     * === PICKLE PREFETCH HINT FOR NON-BUCKET RANKING ===
     *
     * Same indirect access pattern as the bucket version:
     *   work_buff[ key_buff_ptr2[i] ]++
     *
     * key_buff_ptr2 = key_array, iterated sequentially.
     * Values are used as indices into work_buff (= key_buff1 copy).
     * For large MAX_KEY, this causes scattered cache misses.
     */
#if ENABLE_PICKLEDEVICE==1
    if (use_pdev == 1) {
        #pragma omp for nowait schedule(static)
        for( i=0; i<NUM_KEYS; i++ ) {
            if ( i + (INT_TYPE)prefetch_distance < NUM_KEYS ) {
                *UCPage = (uint64_t)(&key_buff_ptr2[i]);
            }
            work_buff[key_buff_ptr2[i]]++;
        }
    } else
#endif
    {
        #pragma omp for nowait schedule(static)
        for( i=0; i<NUM_KEYS; i++ )
            work_buff[key_buff_ptr2[i]]++;
    }

/*  To obtain ranks of each key, successively add the individual key
    population                                          */

    for( i=0; i<MAX_KEY-1; i++ )
        work_buff[i+1] += work_buff[i];

    #pragma omp barrier

/*  Accumulate the global key population */
    for( k=1; k<num_threads; k++ ) {
        #pragma omp for nowait schedule(static)
        for( i=0; i<MAX_KEY; i++ )
            key_buff_ptr[i] += key_buff1_aptr[k][i];
    }

#endif /*USE_BUCKETS*/

#if ENABLE_PICKLEDEVICE==1
    if (PerfPage != NULL)
        *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif

  } /*omp parallel*/

/* This is the partial verify test section */
/* Observe that test_rank_array vals are   */
/* shifted differently for different cases */
    for( i=0; i<TEST_ARRAY_SIZE; i++ )
    {
        k = partial_verify_vals[i];          /* test vals were put here */
        if( 0 < k  &&  k <= NUM_KEYS-1 )
        {
            INT_TYPE key_rank = key_buff_ptr[k-1];
            INT_TYPE test_rank = test_rank_array[i];
            int failed = 0;

            switch( CLASS )
            {
                case 'S':
                    if( i <= 2 )
                        test_rank += iteration;
                    else
                        test_rank -= iteration;
                    break;
                case 'W':
                    if( i < 2 )
                        test_rank += iteration - 2;
                    else
                        test_rank -= iteration;
                    break;
                case 'A':
                    if( i <= 2 )
                        test_rank += iteration - 1;
                    else
                        test_rank -= iteration - 1;
                    break;
                case 'B':
                    if( i == 1 || i == 2 || i == 4 )
                        test_rank += iteration;
                    else
                        test_rank -= iteration;
                    break;
                case 'C':
                    if( i <= 2 )
                        test_rank += iteration;
                    else
                        test_rank -= iteration;
                    break;
                case 'D':
                    if( i < 2 )
                        test_rank += iteration;
                    else
                        test_rank -= iteration;
                    break;
                case 'E':
                    if( i < 2 )
                        test_rank += iteration - 2;
                    else if( i == 2 )
                    {
                        test_rank += iteration - 2;
                        if (iteration > 4)
                            test_rank -= 2;
                        else if (iteration > 2)
                            test_rank -= 1;
                    }
                    else
                        test_rank -= iteration - 2;
                    break;
            }
            if( key_rank != test_rank )
                failed = 1;
            else
                passed_verification++;
            if( failed == 1 )
                printf( "Failed partial verification: "
                        "iteration %d, test key %d\n",
                         iteration, (int)i );
        }
    }


/*  Make copies of rank info for use by full_verify */
    if( iteration == MAX_ITERATIONS )
        key_buff_ptr_global = key_buff_ptr;

}


/*****************************************************************/
/*************             M  A  I  N             ****************/
/*****************************************************************/

int main( int argc, char **argv )
{

    int             i, iteration, timer_on;

    double          timecounter;


/*  Initialize timers  */
    timer_on = check_timer_flag();

    timer_clear( 0 );
    if (timer_on) {
        timer_clear( 1 );
        timer_clear( 2 );
        timer_clear( 3 );
    }

    if (timer_on) timer_start( 3 );


/*  Initialize the verification arrays if a valid class */
    for( i=0; i<TEST_ARRAY_SIZE; i++ )
        switch( CLASS )
        {
            case 'S':
                test_index_array[i] = S_test_index_array[i];
                test_rank_array[i]  = S_test_rank_array[i];
                break;
            case 'A':
                test_index_array[i] = A_test_index_array[i];
                test_rank_array[i]  = A_test_rank_array[i];
                break;
            case 'W':
                test_index_array[i] = W_test_index_array[i];
                test_rank_array[i]  = W_test_rank_array[i];
                break;
            case 'B':
                test_index_array[i] = B_test_index_array[i];
                test_rank_array[i]  = B_test_rank_array[i];
                break;
            case 'C':
                test_index_array[i] = C_test_index_array[i];
                test_rank_array[i]  = C_test_rank_array[i];
                break;
            case 'D':
                test_index_array[i] = D_test_index_array[i];
                test_rank_array[i]  = D_test_rank_array[i];
                break;
            case 'E':
                test_index_array[i] = E_test_index_array[i];
                test_rank_array[i]  = E_test_rank_array[i];
                break;
        };



/*  Printout initial NPB info */
    printf
      ( "\n\n NAS Parallel Benchmarks (NPB3.4-OMP) - IS Benchmark\n\n" );
    printf( " Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS );
    printf( " Iterations:  %d\n", MAX_ITERATIONS );
#ifdef _OPENMP
    printf( " Number of available threads:  %d\n", omp_get_max_threads() );
#endif
    printf( "\n" );

    if (timer_on) timer_start( 1 );

/*  Generate random number sequence and subsequent keys on all procs */
    create_seq( 314159265.00,                    /* Random number gen seed */
                1220703125.00 );                 /* Random number gen mult */

    alloc_key_buff();
    if (timer_on) timer_stop( 1 );


/*  Do one interation for free (i.e., untimed) to guarantee initialization of
    all data and code pages and respective tables */
    rank( 1 );

/*  Start verification counter */
    passed_verification = 0;

    if( CLASS != 'S' ) printf( "\n   iteration\n" );

/*  ============================================================  */
/*  Pickle Prefetcher Setup                                        */
/*  Set up the device AFTER the warmup iteration (trial 0) so     */
/*  that the arrays are initialized and their addresses are final. */
/*  This mirrors the graph benchmarks' two-trial structure:        */
/*    Trial 0 = warmup rank(1) above                               */
/*    Trial 1 = timed iterations below with prefetch hints         */
/*  ============================================================  */
#if ENABLE_GEM5==1
    map_m5_mem();
#endif /* ENABLE_GEM5 */

#if ENABLE_PICKLEDEVICE==1
    wait_till_pickle_device_available();
#endif

#if ENABLE_GEM5==1
    //m5_exit_addr(0);  /* Exit 1: fake exit */
#endif /* ENABLE_GEM5 */

//#if ENABLE_PICKLEDEVICE==1
//    pickle_setup();
//#endif 

#if ENABLE_GEM5==1
    m5_exit_addr(0);  /* Exit 2: the pickle device is turned on after this */
#endif /* ENABLE_GEM5 */

#if ENABLE_PICKLEDEVICE==1
    pickle_setup();
#endif

#if ENABLE_GEM5==1
    m5_exit_addr(0);  /* Exit 3: ROI Start */
#endif /* ENABLE_GEM5 */

/*  Start timer  */
    timer_start( 0 );
    printf("ROI Start\n");

/*  This is the main iteration */
    for( iteration=1; iteration<=MAX_ITERATIONS; iteration++ )
    {
        if( CLASS != 'S' ) printf( "        %d\n", iteration );
        rank( iteration );
    }


/*  End of timing, obtain maximum time of all processors */
    timer_stop( 0 );
    printf("ROI End\n");

#if ENABLE_GEM5==1
    m5_exit_addr(0);  /* Exit 4: ROI End */
#endif /* ENABLE_GEM5 */

    timecounter = timer_read( 0 );


/*  This tests that keys are in sequence: sorting of last ranked key seq
    occurs here, but is an untimed operation                             */
    if (timer_on) timer_start( 2 );
    full_verify();
    if (timer_on) timer_stop( 2 );

    if (timer_on) timer_stop( 3 );

/*  Cleanup Pickle device  */
#if ENABLE_PICKLEDEVICE==1
    pickle_teardown();
#endif

/*  The final printout  */
    if( passed_verification != 5*MAX_ITERATIONS + 1 )
        passed_verification = 0;
    c_print_results( "IS",
                     CLASS,
                     TOTAL_KS1,
                     TOTAL_KS2,
                     0,
                     MAX_ITERATIONS,
                     timecounter,
                     1.0e-6*(double)(TOTAL_KEYS)*MAX_ITERATIONS
                                                  /timecounter,
                     "keys ranked",
                     passed_verification,
                     NPBVERSION,
                     COMPILETIME,
                     CC,
                     CLINK,
                     C_LIB,
                     C_INC,
                     CFLAGS,
                     CLINKFLAGS );


/*  Print additional timers  */
    if (timer_on) {
       double t_total, t_percent;

       t_total = timer_read( 3 );
       printf("\nAdditional timers -\n");
       printf(" Total execution: %8.3f\n", t_total);
       if (t_total == 0.0) t_total = 1.0;
       timecounter = timer_read(1);
       t_percent = timecounter/t_total * 100.;
       printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
       timecounter = timer_read(0);
       t_percent = timecounter/t_total * 100.;
       printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
       timecounter = timer_read(2);
       t_percent = timecounter/t_total * 100.;
       printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);
    }

    return 0;
         /**************************/
}        /*  E N D  P R O G R A M  */
         /**************************/
