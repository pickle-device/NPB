!---------------------------------------------------------------------
!  pickle_ua_mod.f90
!
!  Fortran 90 module for Pickle Prefetcher integration with NPB UA.
!
!  Provides:
!    - iso_c_binding interfaces to the C hooks in pickle_ua_hooks.cpp
!    - Module-level state (device availability, prefetch parameters)
!    - Convenience routines for setup and teardown
!
!  UA's hot indirect-access pattern lives in transf/transfb (transfer.f90)
!  and is structurally a scatter/gather between an element-data array
!  (tx ~ pdiff/pdiffp) and a mortar-data array (tmor ~ pmorx/ppmor),
!  driven by integer index arrays:
!
!      il = idel(i,j,iface,ie)        ! element-collocation index
!      ig = idmo(i,j,je1,je2,iface,ie)! mortar index
!      tx(il)   = ... tmor(ig) ...    ! transf  (gather tmor, scatter tx)
!      tmor(ig) = ... tx(il)  ...     ! transfb (gather tx,   scatter tmor)
!
!  This is structurally identical to the graph-traversal pattern:
!
!      idel/idmo  ~  out_neighbors  (sequential read, integer indices)
!      tx / tmor  ~  node property  (random read/write, double precision)
!
!  Four kernels are registered, covering the diffusion CG inner loop,
!  which is the dominant timed phase of the benchmark:
!
!      Kernel 1 : idel → pdiff   (transf  scatter target)
!      Kernel 2 : idmo → pmorx   (transf  gather  source)
!      Kernel 3 : idel → pdiffp  (transfb gather  source)
!      Kernel 4 : idmo → ppmor   (transfb scatter target)
!
!  The hints fire only when pkl_send_hints is .true., which is set by
!  diffusion() around the CG inner-loop transf/transfb calls.  Other
!  call sites (e.g. transf(tmort,ta1) in the main step loop) leave the
!  flag at .false. so no incorrect prefetches are issued.
!---------------------------------------------------------------------

      module pickle_ua_mod

      use iso_c_binding
      implicit none

!---------------------------------------------------------------------
!  Module-level Pickle state
!---------------------------------------------------------------------
      integer(c_int64_t) :: pkl_use_pdev          = 0
      integer(c_int64_t) :: pkl_prefetch_distance = 0
      integer(c_int)     :: pkl_prefetch_mode     = 0   ! 0=unk,1=single,2=bulk
      integer(c_int64_t) :: pkl_bulk_chunk_size   = 0

! Gate flag: when .true., transf/transfb send prefetch hints.
! Set by diffusion() around the CG inner-loop calls so that hints
! only fire when the specific arrays the kernels were registered for
! (pdiff / pdiffp / pmorx / ppmor) are actually being processed.
      logical            :: pkl_send_hints        = .false.

! Module-level pointers for direct UCPage stores (no function call)
      integer(c_int64_t), pointer, volatile :: pkl_ucpage_kern1 => null()
      integer(c_int64_t), pointer, volatile :: pkl_ucpage_kern2 => null()
      integer(c_int64_t), pointer, volatile :: pkl_ucpage_kern3 => null()
      integer(c_int64_t), pointer, volatile :: pkl_ucpage_kern4 => null()


!---------------------------------------------------------------------
!  iso_c_binding interfaces to C hooks (pickle_ua_hooks.cpp)
!---------------------------------------------------------------------
      interface

         subroutine pickle_ua_init_c(use_pdev, pf_dist,                &
     &                               pf_mode, chunk_sz)                &
     &              bind(C, name='pickle_ua_init')
            import :: c_int64_t, c_int
            integer(c_int64_t), intent(out) :: use_pdev
            integer(c_int64_t), intent(out) :: pf_dist
            integer(c_int),     intent(out) :: pf_mode
            integer(c_int64_t), intent(out) :: chunk_sz
         end subroutine

! Register one (idel-source → element-target) kernel.
!   kernel_id = 1 → idel → pdiff   (transf  scatter)
!   kernel_id = 3 → idel → pdiffp  (transfb gather)
         subroutine pickle_ua_setup_idel_kernel_c(kernel_id,           &
     &              idel_base, idel_n, idel_esz,                       &
     &              tx_base,   tx_n,   tx_esz)                         &
     &              bind(C, name='pickle_ua_setup_idel_kernel')
            import :: c_int, c_int64_t, c_ptr
            integer(c_int),     intent(in) :: kernel_id
            type(c_ptr),        intent(in), value :: idel_base
            integer(c_int64_t), intent(in) :: idel_n, idel_esz
            type(c_ptr),        intent(in), value :: tx_base
            integer(c_int64_t), intent(in) :: tx_n, tx_esz
         end subroutine

! Register one (idmo-source → mortar-target) kernel.
!   kernel_id = 2 → idmo → pmorx   (transf  gather)
!   kernel_id = 4 → idmo → ppmor   (transfb scatter)
         subroutine pickle_ua_setup_idmo_kernel_c(kernel_id,           &
     &              idmo_base, idmo_n, idmo_esz,                       &
     &              tmor_base, tmor_n, tmor_esz)                       &
     &              bind(C, name='pickle_ua_setup_idmo_kernel')
            import :: c_int, c_int64_t, c_ptr
            integer(c_int),     intent(in) :: kernel_id
            type(c_ptr),        intent(in), value :: idmo_base
            integer(c_int64_t), intent(in) :: idmo_n, idmo_esz
            type(c_ptr),        intent(in), value :: tmor_base
            integer(c_int64_t), intent(in) :: tmor_n, tmor_esz
         end subroutine

         subroutine pickle_ua_setup_ucpages_c()                        &
     &              bind(C, name='pickle_ua_setup_ucpages')
         end subroutine

         subroutine pickle_ua_get_ucpage_ptrs_c(p1, p2, p3, p4)        &
     &              bind(C, name='pickle_ua_get_ucpage_ptrs')
            import :: c_ptr
            type(c_ptr), intent(out) :: p1, p2, p3, p4
         end subroutine

         subroutine pickle_ua_perf_start_c(thread_id)                  &
     &              bind(C, name='pickle_ua_perf_start')
            import :: c_int
            integer(c_int), intent(in) :: thread_id
         end subroutine

         subroutine pickle_ua_perf_complete_c(thread_id)               &
     &              bind(C, name='pickle_ua_perf_complete')
            import :: c_int
            integer(c_int), intent(in) :: thread_id
         end subroutine

         subroutine pickle_ua_finalize_c()                             &
     &              bind(C, name='pickle_ua_finalize')
         end subroutine

         subroutine map_m5_mem_c()                                     &
     &              bind(C, name='map_m5_mem_hook')
         end subroutine

         subroutine m5_exit_c()                                        &
     &              bind(C, name='m5_exit_hook')
         end subroutine
      end interface

      contains

!---------------------------------------------------------------------
!  pickle_ua_device_init  —  initialize device and read specs
!---------------------------------------------------------------------
      subroutine pickle_ua_device_init()
         implicit none
         call pickle_ua_init_c(pkl_use_pdev,                           &
     &                         pkl_prefetch_distance,                  &
     &                         pkl_prefetch_mode,                      &
     &                         pkl_bulk_chunk_size)
      end subroutine

      subroutine map_m5_mem()
          implicit none
          call map_m5_mem_c()
      end subroutine

      subroutine m5_exit()
          implicit none
          call m5_exit_c()
      end subroutine

!---------------------------------------------------------------------
!  pickle_ua_setup_ucpage_ptrs  —  convert C UCPage addresses to
!                                  Fortran volatile pointers
!---------------------------------------------------------------------
      subroutine pickle_ua_setup_ucpage_ptrs()
         implicit none
         type(c_ptr) :: cp1, cp2, cp3, cp4
         call pickle_ua_get_ucpage_ptrs_c(cp1, cp2, cp3, cp4)
         call c_f_pointer(cp1, pkl_ucpage_kern1)
         call c_f_pointer(cp2, pkl_ucpage_kern2)
         call c_f_pointer(cp3, pkl_ucpage_kern3)
         call c_f_pointer(cp4, pkl_ucpage_kern4)
      end subroutine

!---------------------------------------------------------------------
!  pickle_ua_device_finalize  —  tear down device
!---------------------------------------------------------------------
      subroutine pickle_ua_device_finalize()
         implicit none
         call pickle_ua_finalize_c()
      end subroutine

      end module pickle_ua_mod
