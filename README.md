NAS Parallel Benchmarks Version 3.4.2 (NPB3.4.2)
--------------------------------------------------

# Compiling workloads

```sh
cd NPB/NPB3.4-OMP/
./compile.sh # a regular version of CG, IS, UA
./compile_sampling.sh # the version of CG, IS, UA that executes according to the sampling points generate sampling points
```

# To generate sampling points

```sh
python3 /workdir/NPB/NPB3.4-OMP/generate_sampling_points.py --output_dir=/workdir/experiments/prefetcher/gem5_configurations/npb_sampling_points/ --workload <workload> --sampling_site=<sampling_site>
```

# Tracking PCs

The `cg.E.sampling.objdump` and `is.D.sampling.objdump` contain the disassembly of the cg/is binaries.
In these files, we annotated the critical loops and identified PCs to be tracked in different sampling sites.

# Other files

- `cg.E.array_data.gz`: the row index of each row of the A matrix; used to determine the number of non-zero elements of each row; which are used to determine the number of warm up rows for each sampling point.
- `is.D.array_data.gz`: the starting index of each bucket; used to determine the number of warm up bucket for each sampling point.
