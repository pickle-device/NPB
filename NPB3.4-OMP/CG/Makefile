SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

BINARY_SUFFFIX=
EXTRA_LIBS =
PICKLE_CXXFLAGS =
PICKLE_LDFLAGS =

#---------------------------------------------------------------------
# Base objects (identical to original)
#---------------------------------------------------------------------
OBJS = cg.o cg_data.o ${COMMON}/print_results.o  \
       ${COMMON}/${RAND}.o ${COMMON}/timers.o ${COMMON}/wtime.o

#---------------------------------------------------------------------
# Pickle Prefetcher objects (conditional)
#
# When ENABLE_PICKLEDEVICE=1:
#   - pickle_cg_mod.o    : Fortran module with iso_c_binding interfaces
#   - pickle_cg_hooks.o  : C++ hooks wrapping the Pickle device API
#   - EXTRA_LIBS          : -lstdc++ for C++ runtime linkage
#   - PICKLE_LDFLAGS      : -lpickledevice (and optional -L path)
#
# The C++ hooks file needs the Pickle device headers.  Set
# PICKLE_INCLUDE_DIR in make.def or on the command line, e.g.:
#   make ENABLE_PICKLEDEVICE=1 PICKLE_INCLUDE_DIR=/path/to/pickle/include
#---------------------------------------------------------------------
ifeq (${ENABLE_GEM5}, 1)
  PICKLE_CXXFLAGS += -DENABLE_GEM5=1 -I${M5OPS_HEADER_PATH} -I${M5_BUILD_PATH}/../../../m5/src/
  PICKLE_LDFLAGS += -L${M5_BUILD_PATH}/out/ -lm5
  BINARY_SUFFIX:=$(BINARY_SUFFIX).m5
endif
ifeq (${ENABLE_PICKLEDEVICE}, 1)
  OBJS += pickle_cg_mod.o pickle_cg_hooks.o
  EXTRA_LIBS += -lstdc++
  PICKLE_CXXFLAGS += -DENABLE_PICKLEDEVICE=1
  PICKLE_LDFLAGS += -lpickledevice
  BINARY_SUFFIX:=$(BINARY_SUFFIX).pdev
endif

include ../sys/make.common
.SUFFIXES: .c .f .f90 .h .o .cpp

#---------------------------------------------------------------------
# Link
#---------------------------------------------------------------------
${PROGRAM}: config ${OBJS}
	${FLINK} ${FLINKFLAGS} -o ${PROGRAM}${BINARY_SUFFIX} ${OBJS} ${F_LIB} ${EXTRA_LIBS} ${PICKLE_LDFLAGS}

#---------------------------------------------------------------------
# Fortran compile rule  (passes -cpp for preprocessor directives)
#---------------------------------------------------------------------
.f90.o:
ifeq (${ENABLE_PICKLEDEVICE}, 1)
	${FCOMPILE} -DENABLE_PICKLEDEVICE=1 -DENABLE_GEM5=1 -cpp $<
else
	${FCOMPILE} $<
endif

#---------------------------------------------------------------------
# C++ compile rule for the Pickle hooks
#---------------------------------------------------------------------
pickle_cg_hooks.o: pickle_cg_hooks.cpp
	${CXX} -c ${CFLAGS} ${PICKLE_CXXFLAGS} pickle_cg_hooks.cpp

#---------------------------------------------------------------------
# Dependency chain
#
# Module compilation order matters in Fortran:
#   1. cg_data.o       — defines cg_data module (used by everything)
#   2. pickle_cg_mod.o — defines pickle_cg_mod module (uses iso_c_binding)
#   3. cg.o            — uses cg_data AND pickle_cg_mod
#
# pickle_cg_hooks.o has no Fortran module dependencies.
#---------------------------------------------------------------------
cg_data.o:		cg_data.f90 npbparams.h

ifeq (${ENABLE_PICKLEDEVICE}, 1)
pickle_cg_mod.o:	pickle_cg_mod.f90
cg.o:			cg.f90 cg_data.o pickle_cg_mod.o
else
cg.o:			cg.f90 cg_data.o
endif

clean:
	- rm -f *.o *~ *.mod
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
