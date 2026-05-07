SHELL=/bin/sh
BENCHMARK=ua
BENCHMARKU=UA
UPDATE=$(VERSION)
UXT=

include ../config/make.def

BINARY_SUFFIX=
EXTRA_LIBS =
PICKLE_CXXFLAGS =
PICKLE_LDFLAGS =

#---------------------------------------------------------------------
# Base objects
#---------------------------------------------------------------------
OBJS = ua.o ua_data.o convect.o diffuse.o adapt.o move.o mason.o \
       precond.o utils.o verify.o setup.o transfer$(UXT).o \
       ${COMMON}/print_results.o ${COMMON}/timers.o ${COMMON}/wtime.o

ifeq (${M5_ANNOTATION}, 1)
	OBJS += ${COMMON}/m5_interface.o ${COMMON}/m5op.o
endif

#---------------------------------------------------------------------
# Pickle Prefetcher objects (conditional, mirrors CG benchmark)
#
# When ENABLE_PICKLEDEVICE=1:
#   - pickle_ua_mod.o    : Fortran module with iso_c_binding interfaces
#   - pickle_ua_hooks.o  : C++ hooks wrapping the Pickle device API
#   - EXTRA_LIBS         : -lstdc++ for C++ runtime linkage
#   - PICKLE_LDFLAGS     : -lpickledevice
#---------------------------------------------------------------------
ifeq (${ENABLE_GEM5}, 1)
  PICKLE_CXXFLAGS += -DENABLE_GEM5=1 -I${M5OPS_HEADER_PATH} -I${M5_BUILD_PATH}/../../../m5/src/
  PICKLE_LDFLAGS += -L${M5_BUILD_PATH}/out/ -lm5
  BINARY_SUFFIX:=$(BINARY_SUFFIX).m5
endif
ifeq (${ENABLE_PICKLEDEVICE}, 1)
  OBJS += pickle_ua_mod.o pickle_ua_hooks.o
  EXTRA_LIBS += -lstdc++
  PICKLE_CXXFLAGS += -DENABLE_PICKLEDEVICE=1
  PICKLE_LDFLAGS += -lpickledevice
  BINARY_SUFFIX:=$(BINARY_SUFFIX).pdev
endif

include ../sys/make.common
.SUFFIXES: .c .f .f90 .h .o .cpp

# npbparams.h is included by ua_data module (via ua_data.o)

${PROGRAM}: config ${OBJS}
	@if [ x$(UPDATE) = xau -o x$(UPDATE) = xAU ] ; then	\
		${MAKE} UXT=_au ua-def;	\
	elif [ x$(UPDATE) = xrd -o x$(UPDATE) = xRD ] ; then	\
		${MAKE} UXT=_rd ua-rd;	\
	else				\
		${MAKE} ua-def;		\
	fi

ua-def: ${OBJS}
	${FLINK} ${FLINKFLAGS} -o ${PROGRAM}${BINARY_SUFFIX} ${OBJS} ${F_LIB} ${EXTRA_LIBS} ${PICKLE_LDFLAGS}

ua-rd: ${OBJS} tmorwork.o
	${FLINK} ${FLINKFLAGS} -o ${PROGRAM}${BINARY_SUFFIX} ${OBJS} tmorwork.o ${F_LIB} ${EXTRA_LIBS} ${PICKLE_LDFLAGS}

#---------------------------------------------------------------------
# Fortran compile rule  (passes -cpp for preprocessor directives when
# Pickle is enabled, so #if ENABLE_PICKLEDEVICE/#if ENABLE_GEM5 work)
#---------------------------------------------------------------------
.f90.o:
ifeq (${ENABLE_PICKLEDEVICE}, 1)
	${FCOMPILE} -DENABLE_PICKLEDEVICE=1 -DENABLE_GEM5=${ENABLE_GEM5} -cpp $<
else
ifeq (${M5_ANNOTATION}, 1)
	${FCOMPILE} -DM5_ANNOTATION $<
else
	${FCOMPILE} $<
endif
endif

#---------------------------------------------------------------------
# C++ compile rule for the Pickle hooks
#---------------------------------------------------------------------
pickle_ua_hooks.o: pickle_ua_hooks.cpp
	${CXX} -c ${CFLAGS} ${PICKLE_CXXFLAGS} pickle_ua_hooks.cpp

#---------------------------------------------------------------------
# Dependency chain
#
# Module compilation order matters in Fortran:
#   1. ua_data.o        — defines ua_data module (used by everything)
#   2. pickle_ua_mod.o  — defines pickle_ua_mod module (uses iso_c_binding)
#   3. transfer.o, diffuse.o, ua.o — use both modules
#
# pickle_ua_hooks.o has no Fortran module dependencies.
#---------------------------------------------------------------------
ua_data.o:   ua_data.f90  npbparams.h

ifeq (${ENABLE_PICKLEDEVICE}, 1)
pickle_ua_mod.o: pickle_ua_mod.f90
ua.o:        ua.f90       ua_data.o pickle_ua_mod.o
setup.o:     setup.f90    ua_data.o
convect.o:   convect.f90  ua_data.o
adapt.o:     adapt.f90    ua_data.o
move.o:      move.f90     ua_data.o
diffuse.o:   diffuse.f90  ua_data.o pickle_ua_mod.o
mason.o:     mason.f90    ua_data.o
precond.o:   precond.f90  ua_data.o
transfer.o:  transfer.f90 ua_data.o pickle_ua_mod.o
transfer_au.o:  transfer_au.f90 ua_data.o pickle_ua_mod.o
transfer_rd.o:  transfer_rd.f90 ua_data.o pickle_ua_mod.o tmorwork.o
utils.o:     utils.f90    ua_data.o
verify.o:    verify.f90   ua_data.o
else
ua.o:        ua.f90       ua_data.o
setup.o:     setup.f90    ua_data.o
convect.o:   convect.f90  ua_data.o
adapt.o:     adapt.f90    ua_data.o
move.o:      move.f90     ua_data.o
diffuse.o:   diffuse.f90  ua_data.o
mason.o:     mason.f90    ua_data.o
precond.o:   precond.f90  ua_data.o
transfer.o:  transfer.f90 ua_data.o
transfer_au.o:  transfer_au.f90 ua_data.o
transfer_rd.o:  transfer_rd.f90 ua_data.o tmorwork.o
utils.o:     utils.f90    ua_data.o
verify.o:    verify.f90   ua_data.o
endif
tmorwork.o:  tmorwork.f90

clean:
	- rm -f *.o *~ *.mod mputil*
	- rm -f npbparams.h core
