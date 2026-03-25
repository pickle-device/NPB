!---------------------------------------------------------------------
!  pickle_cg_mod.f90
!
!  Fortran 90 module for Pickle Prefetcher integration with NPB CG.
!
!  Provides:
!    - iso_c_binding interfaces to the C hooks in pickle_cg_hooks.cpp
!    - Module-level state (device availability, prefetch parameters)
!    - Convenience routines for setup and teardown
!
!  The SpMV kernel in CG has the CSR indirect access pattern:
!
!      suml = suml + a(k) * p( colidx(k) )
!                            ^^^^^^^^^^^^^^^^
!                            indirect: sequential colidx → random p
!
!  which maps to the graph traversal pattern:
!
!      rowstr  ≡  out_index      (row pointers)
!      colidx  ≡  out_neighbors  (column indices)
!      p / z   ≡  node property  (dense target vector)
!
!---------------------------------------------------------------------

      module pickle_cg_mod

      use iso_c_binding
      implicit none

!---------------------------------------------------------------------
!  Module-level Pickle state
!---------------------------------------------------------------------
      integer(c_int64_t) :: pkl_use_pdev          = 0
      integer(c_int64_t) :: pkl_prefetch_distance = 0
      integer(c_int)     :: pkl_prefetch_mode     = 0   ! 0=unk,1=single,2=bulk
      integer(c_int64_t) :: pkl_bulk_chunk_size   = 0

!---------------------------------------------------------------------
!  iso_c_binding interfaces to C hooks (pickle_cg_hooks.cpp)
!
!  All parameters are passed by reference (Fortran default) unless
!  marked VALUE.  The C functions dereference pointers accordingly.
!---------------------------------------------------------------------
      interface

         subroutine pickle_cg_init_c(use_pdev, pf_dist,              &
     &                               pf_mode, chunk_sz)              &
     &              bind(C, name='pickle_cg_init')
            import :: c_int64_t, c_int
            integer(c_int64_t), intent(out) :: use_pdev
            integer(c_int64_t), intent(out) :: pf_dist
            integer(c_int),     intent(out) :: pf_mode
            integer(c_int64_t), intent(out) :: chunk_sz
         end subroutine

         subroutine pickle_cg_setup_spmv_c(kernel_id,                &
     &              rowstr_base, rowstr_n, rowstr_esz,               &
     &              colidx_base, colidx_n, colidx_esz,              &
     &              vec_base,    vec_n,    vec_esz)                  &
     &              bind(C, name='pickle_cg_setup_spmv')
            import :: c_int, c_int64_t, c_ptr
            integer(c_int),     intent(in) :: kernel_id
            type(c_ptr),        intent(in), value :: rowstr_base
            integer(c_int64_t), intent(in) :: rowstr_n, rowstr_esz
            type(c_ptr),        intent(in), value :: colidx_base
            integer(c_int64_t), intent(in) :: colidx_n, colidx_esz
            type(c_ptr),        intent(in), value :: vec_base
            integer(c_int64_t), intent(in) :: vec_n,    vec_esz
         end subroutine

         subroutine pickle_cg_setup_ucpages_c()                      &
     &              bind(C, name='pickle_cg_setup_ucpages')
         end subroutine

         subroutine pickle_cg_spmv_hint_c(kernel_id, row_0based)     &
     &              bind(C, name='pickle_cg_spmv_hint')
            import :: c_int, c_int64_t
            integer(c_int),     intent(in) :: kernel_id
            integer(c_int64_t), intent(in) :: row_0based
         end subroutine

         subroutine pickle_cg_perf_start_c(thread_id)                &
     &              bind(C, name='pickle_cg_perf_start')
            import :: c_int
            integer(c_int), intent(in) :: thread_id
         end subroutine

         subroutine pickle_cg_perf_complete_c(thread_id)             &
     &              bind(C, name='pickle_cg_perf_complete')
            import :: c_int
            integer(c_int), intent(in) :: thread_id
         end subroutine

         subroutine pickle_cg_finalize_c()                           &
     &              bind(C, name='pickle_cg_finalize')
         end subroutine

      end interface

      contains

!---------------------------------------------------------------------
!  pickle_cg_device_init  —  initialize device and read specs
!---------------------------------------------------------------------
      subroutine pickle_cg_device_init()
         implicit none
         call pickle_cg_init_c(pkl_use_pdev,                          &
     &                         pkl_prefetch_distance,                 &
     &                         pkl_prefetch_mode,                     &
     &                         pkl_bulk_chunk_size)
      end subroutine

!---------------------------------------------------------------------
!  pickle_cg_device_finalize  —  tear down device
!---------------------------------------------------------------------
      subroutine pickle_cg_device_finalize()
         implicit none
         call pickle_cg_finalize_c()
      end subroutine

      end module pickle_cg_mod
