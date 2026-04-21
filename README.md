NAS Parallel Benchmarks Version 3.4.2 (NPB3.4.2)
--------------------------------------------------

  NAS Parallel Benchmarks Team
  NASA Ames Research Center
  Moffett Field, CA   94035-1000

  E-mail:  npb@nas.nasa.gov                                      
  Fax:     (650) 604-3957                                        
  http://www.nas.nasa.gov/Software/NPB/

--------------------------------------------------

This repo adds m5 annotations to the NAS Parallel Benchmarks located in
`NPB/NPB3.4-OMP`.

The annotations are added to the timed sections as follows,
- An `m5_work_begin()` call before `timer_start(t_bench)`, or
`timer_start(t_total)`, or an equivalent.
- An `m5_work_end()` call after `timer_stop(t_bench)`, or `timer_stop(t_total)`, 
or an equivalent.

**Notes:**
- For the `DC` benchmark, the work is split into tasks via OMP directive.
Each task uses a different timer for its timed section. To keep it simple,
we decided to put an `m5_work_begin()` before the OMP task directive, and
another `m5_work_end()` after the OMP task directive.

In order to compile the benchmarks with m5 annotations, the
following environment variables must be set,

```sh
M5_ANNOTATION # If M5_ANNOTATION=1, the build system will compile the binaries
              # with m5 annotations.

GEM5_INCLUDE_DIR # Path to the gem5's include/ directory.
                 # Typically, the path is gem5/include.

GEM5_M5_ABI_DIR # Path to m5 utility abi folder for a specific ISA.
                # Typically,
                # For x86: gem5/util/m5/src/abi/x86/
                # For arm64: gem5/util/m5/src/abi/arm64/
                # For riscv: gm5/util/m5/src/abi/riscv
```

If you're cross compiling, you might want to set additional environment
variables as follows,

```sh
FC # Path to the gfortran compiler for compiling **guest** binaries.
CC # Path to the gcc compiler for compiling **guest** binaries.
UCC # Path to the gcc compiler for compiling utilities **guest** objects.
SYS_UCC # Path to the gcc compiler for compiling utilities **host** objects.
```
