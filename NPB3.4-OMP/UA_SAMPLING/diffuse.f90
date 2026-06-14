!---------------------------------------------------------------------
      subroutine diffusion(ifmortar,sampling_site,          &
     &  current_ua_step,ua_starting_step,cg_starting_iter,    &
     &  transfer_starting_element,transfer_num_warmup_elements)
!---------------------------------------------------------------------
!     advance the diffusion term using CG iterations
!---------------------------------------------------------------------

      use ua_data
      use iso_c_binding

#if ENABLE_PICKLEDEVICE==1
      use pickle_ua_mod
#endif
      implicit none

      double precision  rho_aux, rho1, rho2, beta, cona
      logical ifmortar
      integer iter,ie, im,iside,i,j,k

      integer, intent(in) :: sampling_site
      integer, intent(in) :: current_ua_step
      integer, intent(in) :: ua_starting_step
      integer, intent(in) :: cg_starting_iter
      integer, intent(in) :: transfer_starting_element
      integer, intent(in) :: transfer_num_warmup_elements

#if ENABLE_PICKLEDEVICE==1
      integer(c_int)     :: pkl_kid
      integer(c_int64_t) :: pkl_idel_n, pkl_idel_esz
      integer(c_int64_t) :: pkl_idmo_n, pkl_idmo_esz
      integer(c_int64_t) :: pkl_pdiff_n, pkl_pdiff_esz
      integer(c_int64_t) :: pkl_pmor_n,  pkl_pmor_esz
#endif

      if (timeron) call timer_start(t_diffusion)
!.....set up diagonal preconditioner
      if (ifmortar) then
        call setuppc
        call setpcmo
      end if

!.....arrays t and umor are accumlators of (am pm) in the CG algorithm
!     (see the specification)

      call r_init_omp(t,ntot,0.d0)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
      do i=1,nmor
        umor(i)=0.d0
      end do
!$OMP END PARALLEL DO

!.....calculate initial am (see specification) in CG algorithm

!.....trhs and rmor are combined to generate r0 in CG algorithm.
!     pdiff and pmorx are combined to generate q0 in the CG algorithm.
!     rho1 is  (qm,rm) in the CG algorithm.

      rho1 = 0.d0
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(im,ie,i,j,k) REDUCTION(+:rho1)
!$OMP DO
       do ie=1,nelt
         do k=1,lx1
           do j=1,lx1
             do i=1,lx1
               pdiff(i,j,k,ie) = dpcelm(i,j,k,ie)*trhs(i,j,k,ie)
               rho1            = rho1 + trhs(i,j,k,ie)*pdiff(i,j,k,ie)*  &
     &                                          tmult(i,j,k,ie)
             end do
           end do
         end do
       end do
!$OMP END DO nowait

!$OMP DO
      do im = 1, nmor
        pmorx(im) = dpcmor(im)*rmor(im)
        rho1      = rho1 + rmor(im)*pmorx(im)
      end do
!$OMP END DO nowait
!$OMP END PARALLEL

!.................................................................
!     commence conjugate gradient iteration
!.................................................................

      do iter=1, nmxh
        if(iter.gt.1) then
          rho_aux = 0.d0
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(im,ie,i,j,k) REDUCTION(+:rho_aux)
!$OMP DO
!.........pdiffp and ppmor are combined to generate q_m+1 in the specification
!         rho_aux is (q_m+1,r_m+1)
          do ie = 1, nelt
            do k=1,lx1
              do j=1,lx1
                do i=1,lx1
                  pdiffp(i,j,k,ie) = dpcelm(i,j,k,ie)*trhs(i,j,k,ie)
                  rho_aux =rho_aux+trhs(i,j,k,ie)*pdiffp(i,j,k,ie)*  &
     &                                            tmult(i,j,k,ie)
                end do
              end do
            end do
          end do
!$OMP END DO nowait
!$OMP DO
          do im = 1, nmor
            ppmor(im) = dpcmor(im)*rmor(im)
            rho_aux = rho_aux + rmor(im)*ppmor(im)
          end do
!$OMP END DO nowait
!$OMP END PARALLEL

!.........compute bm (beta) in the specification
          rho2 = rho1
          rho1 = rho_aux
          beta = rho1/rho2
!.........update p_m+1 in the specification
          call adds1m1(pdiff, pdiffp, beta,ntot)
          call adds1m1(pmorx, ppmor,  beta, nmor)
        end if

!.......compute matrix vector product: (theta pm) in the specification

        if (timeron) call timer_start(t_transf)
        if (sampling_site .eq. 1 .and. current_ua_step .eq. ua_starting_step .and. iter .eq. cg_starting_iter) then
           call transf_some_elements(pmorx,pdiff,1,transfer_starting_element-1)
           !------------- Exit 1: switch CPUs -------------
#if ENABLE_GEM5==1
            !$omp barrier
            !$omp master
                write(*,*) 'sampling site 1, ua_step = ',     &
     &            current_ua_step, 'cg_iter = ', iter,        &
     &            ' warmup: [', transfer_starting_element,    &
     &            '...',                                      &
     &            transfer_starting_element                   &
     &              +transfer_num_warmup_elements-1, ']'
                call map_m5_mem()
                call m5_exit()
            !$omp end master
            !$omp barrier
#endif
           ! warmup the cache
           call transf_some_elements(pmorx,pdiff,transfer_starting_element,transfer_starting_element+transfer_num_warmup_elements-1)
           !------------- Exit 2: done warmup cache, now setup Pickle device -------------
#if ENABLE_GEM5==1
           !$omp barrier
           !$omp master
               write(*,*) 'sampling site 1: done warm up'
               call m5_exit()
           !$omp end master
           !$omp barrier
#endif
#if ENABLE_PICKLEDEVICE==1
           !$omp barrier
           !$omp master
            write(*,*) 'sampling site 1: setup pickle device'
            call pickle_ua_device_init()

            if (pkl_use_pdev .eq. 1) then

              pkl_idel_n    = int(size(idel),  c_int64_t)
              pkl_idel_esz  = int(storage_size(idel(1,1,1,1))/8,        &
      &                            c_int64_t)
              pkl_idmo_n    = int(size(idmo),  c_int64_t)
              pkl_idmo_esz  = int(storage_size(idmo(1,1,1,1,1,1))/8,    &
      &                            c_int64_t)
              pkl_pdiff_n   = int(size(pdiff), c_int64_t)
              pkl_pdiff_esz = int(storage_size(pdiff(1,1,1,1))/8,       &
      &                            c_int64_t)
              pkl_pmor_n    = int(size(pmorx), c_int64_t)
              pkl_pmor_esz  = int(storage_size(pmorx(1))/8, c_int64_t)

              ! Kernel 1: idel → pdiff   (transf  scatter target)
              pkl_kid = 1
              call pickle_ua_setup_idel_kernel_c(pkl_kid,               &
      &            c_loc(idel(1,1,1,1)),  pkl_idel_n,  pkl_idel_esz,  &
      &            c_loc(pdiff(1,1,1,1)), pkl_pdiff_n, pkl_pdiff_esz)

              ! Kernel 2: idmo → pmorx   (transf  gather  source)
              pkl_kid = 2
              call pickle_ua_setup_idmo_kernel_c(pkl_kid,               &
      &            c_loc(idmo(1,1,1,1,1,1)), pkl_idmo_n, pkl_idmo_esz,&
      &            c_loc(pmorx(1)),          pkl_pmor_n, pkl_pmor_esz)

              ! Kernel 3: idel → pdiffp  (transfb gather  source)
              pkl_kid = 3
              call pickle_ua_setup_idel_kernel_c(pkl_kid,               &
      &            c_loc(idel(1,1,1,1)),   pkl_idel_n,  pkl_idel_esz, &
      &            c_loc(pdiffp(1,1,1,1)), pkl_pdiff_n, pkl_pdiff_esz)

              ! Kernel 4: idmo → ppmor   (transfb scatter target)
              pkl_kid = 4
              call pickle_ua_setup_idmo_kernel_c(pkl_kid,               &
      &            c_loc(idmo(1,1,1,1,1,1)), pkl_idmo_n, pkl_idmo_esz,&
      &            c_loc(ppmor(1)),          pkl_pmor_n, pkl_pmor_esz)

              ! Register num elements update kernel
              call pickle_ua_num_elements_update_kernel_c()

              ! Obtain UCPage communication area
              call pickle_ua_setup_ucpages_c()
              call pickle_ua_setup_ucpage_ptrs()
          endif
            !$omp end master
            !$omp barrier
#endif
           !------------- Exit 3: done setting up Pickle device; now start sampling -------------
#if ENABLE_GEM5==1
           !$omp barrier
           !$omp master
               write(*,*) 'sampling site 1: ROI Start; starting from ', transfer_starting_element+transfer_num_warmup_elements
               call m5_exit()
           !$omp end master
           !$omp barrier
#endif
#if ENABLE_PICKLEDEVICE==1
           if (pkl_use_pdev .eq. 1) then
              call transf_some_elements_with_pdev(pmorx,pdiff,transfer_starting_element+transfer_num_warmup_elements,nelt)
           else
              call transf_some_elements(pmorx,pdiff,transfer_starting_element+transfer_num_warmup_elements,nelt)
           end if
#else
           call transf_some_elements(pmorx,pdiff,transfer_starting_element+transfer_num_warmup_elements,nelt)
#endif

        else
           call transf(pmorx,pdiff)
        end if
        !------------- Exit 4: simulation should have exited at this point; but if not, exit now -------------
#if ENABLE_GEM5==1
            !$omp barrier
            !$omp master
            if (sampling_site .eq. 1 .and. current_ua_step .eq. ua_starting_step .and. iter .eq. cg_starting_iter) then
              write(*,*) 'Exiting simulation as it should have exited at this point; but if not, exit now'
              call m5_exit()
            end if
            !$omp end master
            !$omp barrier
#endif
        if (timeron) call timer_stop(t_transf)

!.......compute pdiffp which is (A theta pm) in the specification
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ie)
        do ie=1, nelt
          call laplacian(pdiffp(1,1,1,ie),pdiff(1,1,1,ie),size_e(ie))
        end do
!$OMP END PARALLEL DO

!.......compute ppmor which will be used to compute (thetaT A theta pm)
!       in the specification
        if (timeron) call timer_start(t_transfb)
        if (sampling_site .eq. 2 .and. current_ua_step .eq. ua_starting_step .and. iter .eq. cg_starting_iter) then
           call transfb_some_elements(ppmor,pdiffp,1,transfer_starting_element-1)
           !------------- Exit 1: switch CPUs -------------
#if ENABLE_GEM5==1
            !$omp barrier
            !$omp master
                write(*,*) 'sampling site 2, ua_step = ',     &
     &            current_ua_step, 'cg_iter = ', iter,        &
     &            ' warmup: [', transfer_starting_element,    &
     &            '...',                                      &
     &            transfer_starting_element                   &
     &              +transfer_num_warmup_elements-1, ']'
                call map_m5_mem()
                call m5_exit()
            !$omp end master
            !$omp barrier
#endif
           ! warmup the cache
           call transfb_some_elements(ppmor,pdiffp,           &
     &       transfer_starting_element,                       &
     &       transfer_starting_element                        &
     &         +transfer_num_warmup_elements-1)
           !------------- Exit 2: done warmup cache, now setup Pickle device -------------
#if ENABLE_GEM5==1
           !$omp barrier
           !$omp master
               write(*,*) 'sampling site 2: done warm up'
               call m5_exit()
           !$omp end master
           !$omp barrier
#endif
#if ENABLE_PICKLEDEVICE==1
           !$omp barrier
           !$omp master
            write(*,*) 'sampling site 2: setup pickle device'
            call pickle_ua_device_init()

            if (pkl_use_pdev .eq. 1) then

              pkl_idel_n    = int(size(idel),  c_int64_t)
              pkl_idel_esz  = int(storage_size(idel(1,1,1,1))/8,        &
      &                            c_int64_t)
              pkl_idmo_n    = int(size(idmo),  c_int64_t)
              pkl_idmo_esz  = int(storage_size(idmo(1,1,1,1,1,1))/8,    &
      &                            c_int64_t)
              pkl_pdiff_n   = int(size(pdiff), c_int64_t)
              pkl_pdiff_esz = int(storage_size(pdiff(1,1,1,1))/8,       &
      &                            c_int64_t)
              pkl_pmor_n    = int(size(pmorx), c_int64_t)
              pkl_pmor_esz  = int(storage_size(pmorx(1))/8, c_int64_t)

              ! Kernel 1: idel → pdiff   (transf  scatter target)
              pkl_kid = 1
              call pickle_ua_setup_idel_kernel_c(pkl_kid,               &
      &            c_loc(idel(1,1,1,1)),  pkl_idel_n,  pkl_idel_esz,  &
      &            c_loc(pdiff(1,1,1,1)), pkl_pdiff_n, pkl_pdiff_esz)

              ! Kernel 2: idmo → pmorx   (transf  gather  source)
              pkl_kid = 2
              call pickle_ua_setup_idmo_kernel_c(pkl_kid,               &
      &            c_loc(idmo(1,1,1,1,1,1)), pkl_idmo_n, pkl_idmo_esz,&
      &            c_loc(pmorx(1)),          pkl_pmor_n, pkl_pmor_esz)

              ! Kernel 3: idel → pdiffp  (transfb gather  source)
              pkl_kid = 3
              call pickle_ua_setup_idel_kernel_c(pkl_kid,               &
      &            c_loc(idel(1,1,1,1)),   pkl_idel_n,  pkl_idel_esz, &
      &            c_loc(pdiffp(1,1,1,1)), pkl_pdiff_n, pkl_pdiff_esz)

              ! Kernel 4: idmo → ppmor   (transfb scatter target)
              pkl_kid = 4
              call pickle_ua_setup_idmo_kernel_c(pkl_kid,               &
      &            c_loc(idmo(1,1,1,1,1,1)), pkl_idmo_n, pkl_idmo_esz,&
      &            c_loc(ppmor(1)),          pkl_pmor_n, pkl_pmor_esz)

              ! Register num elements update kernel
              call pickle_ua_num_elements_update_kernel_c()

              ! Obtain UCPage communication area
              call pickle_ua_setup_ucpages_c()
              call pickle_ua_setup_ucpage_ptrs()
            endif
            !$omp end master
            !$omp barrier
#endif
           !------------- Exit 3: done setting up Pickle device; now start sampling -------------
#if ENABLE_GEM5==1
           !$omp barrier
           !$omp master
               write(*,*) 'sampling site 3: ROI Start; starting from ', transfer_starting_element+transfer_num_warmup_elements
               call m5_exit()
           !$omp end master
           !$omp barrier
#endif
#if ENABLE_PICKLEDEVICE==1
           if (pkl_use_pdev .eq. 1) then
              call transfb_some_elements_with_pdev(ppmor,pdiffp,transfer_starting_element+transfer_num_warmup_elements,nelt)
           else
              call transfb_some_elements(ppmor,pdiffp,transfer_starting_element+transfer_num_warmup_elements,nelt)
           end if
#else
           call transfb_some_elements(ppmor,pdiffp,transfer_starting_element+transfer_num_warmup_elements,nelt)
#endif

        else
           call transfb(ppmor,pdiffp)
        end if
        !------------- Exit 4: simulation should have exited at this point; but if not, exit now -------------
#if ENABLE_GEM5==1
            !$omp barrier
            !$omp master
            if (sampling_site .eq. 2 .and. current_ua_step .eq. ua_starting_step .and. iter .eq. cg_starting_iter) then
              write(*,*) 'Exiting simulation as it should have exited at this point; but if not, exit now'
              call m5_exit()
            end if
            !$omp end master
            !$omp barrier
#endif
        if (timeron) call timer_stop(t_transfb)

!.......apply boundary condition
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(ie,iside)
        do ie=1,nelt
          do iside=1,nsides
            if(cbc(iside,ie).eq.0)then
              call facev(pdiffp(1,1,1,ie),iside,0.d0)
            end if
          end do
        end do
!$OMP END PARALLEL DO

!.......compute cona which is (pm,theta T A theta pm)
        cona = 0.d0
!$OMP PARALLEL DEFAULT(SHARED) PRIVATE(im,ie,i,j,k) REDUCTION(+:cona)
!$OMP DO
        do ie = 1, nelt
          do k=1,lx1
            do j=1,lx1
              do i=1,lx1
                cona = cona +  &
     &          pdiff(i,j,k,ie)*pdiffp(i,j,k,ie)*tmult(i,j,k,ie)
              end do
             end do
          end do
        end do
!$OMP END DO nowait
!$OMP DO
        do im = 1, nmor
          ppmor(im) = ppmor(im)*tmmor(im)
          cona = cona + pmorx(im)*ppmor(im)
        end do
!$OMP END DO nowait
!$OMP END PARALLEL

!.......compute am
        cona = rho1/cona
!.......compute (am pm)
        call adds2m1(t,    pdiff,   cona, ntot)
        call adds2m1(umor, pmorx,   cona, nmor)
!.......compute r_m+1
        call adds2m1(trhs, pdiffp, -cona, ntot)
        call adds2m1(rmor, ppmor,  -cona, nmor)

      end do

      if (timeron) call timer_start(t_transf)
      call transf(umor,t)
      if (timeron) call timer_stop(t_transf)
      if (timeron) call timer_stop(t_diffusion)

      return
      end


!------------------------------------------------------------------
      subroutine laplacian(r,u,sizei)
!------------------------------------------------------------------
!     compute  r = visc*[A]x +[B]x on a given element.
!------------------------------------------------------------------

      use ua_data
      implicit none

      double precision r(lx1,lx1,lx1), u(lx1,lx1,lx1), rdtime
      integer i,j,k, ix,iz, sizei

      double precision tm1(lx1,lx1,lx1),tm2(lx1,lx1,lx1)

      rdtime = 1.d0/dtime

      call r_init(tm1,nxyz,0.d0)
      do iz=1,lx1
        do k = 1, lx1
          do j = 1, lx1
            do i = 1, lx1
              tm1(i,j,iz) = tm1(i,j,iz)+wdtdr(i,k)*u(k,j,iz)
            end do
          end do
        end do
      end do

      call r_init(tm2,nxyz,0.d0)
      do iz=1,lx1
        do k = 1, lx1
          do j = 1, lx1
            do i = 1, lx1
              tm2(i,j,iz) = tm2(i,j,iz)+u(i,k,iz)*wdtdr(k,j)
            end do
          end do
        end do
      end do

      call r_init(r,nxyz,0.d0)
      do k = 1, lx1
        do iz=1, lx1
          do j = 1, lx1
            do i = 1, lx1
              r(i,j,iz) = r(i,j,iz)+u(i,j,k)*wdtdr(k,iz)
            end do
          end do
        end do
      end do

!.....collocate with remaining weights and sum to complete factorization.

!      do ix=1,nxyz
!         r(ix,1,1)=visc*(tm1(ix,1,1)*g4m1_s(ix,1,1,sizei)+
!     &                   tm2(ix,1,1)*g5m1_s(ix,1,1,sizei)+
!     &                     r(ix,1,1)*g6m1_s(ix,1,1,sizei))+
!     &               bm1_s(ix,1,1,sizei)*rdtime*u(ix,1,1)
!      end do
      do k=1,lx1
        do j=1,lx1
          do i=1,lx1
            r(i,j,k)=visc*(tm1(i,j,k)*g4m1_s(i,j,k,sizei)+  &
     &                   tm2(i,j,k)*g5m1_s(i,j,k,sizei)+  &
     &                    r(i,j,k)*g6m1_s(i,j,k,sizei))+  &
     &               bm1_s(i,j,k,sizei)*rdtime*u(i,j,k)
          end do
        end do
      end do

      return
      end



