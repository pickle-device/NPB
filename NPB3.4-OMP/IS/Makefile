SHELL=/bin/sh
BENCHMARK=is
BENCHMARKU=IS

include ../config/make.def

include ../sys/make.common

OBJS = is.o \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o

ifeq (${ENABLE_PICKLEDEVICE}, 1)
	CXX_FLAGS += -I$(M5OPS_HEADER_PATH) -I$(M5OPS_HEADER_PATH)/../util/m5/src
	CXX_FLAGS += -DENABLE_GEM5=1
	LDFLAGS   += -lm5 -L$(M5_BUILD_PATH)/out/
	LDFLAGS   += -DENABLE_GEM5=1
endif

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${CXX_FLAGS} ${LDFLAGS}

.c.o:
ifeq (${ENABLE_PICKLEDEVICE}, 1)
	${CCOMPILE} ${CXX_FLAGS} ${LDFLAGS} $<
else
	${CCOMPILE} $<
endif

is.o:             is.c  npbparams.h


clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi
