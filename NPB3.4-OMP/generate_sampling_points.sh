#!/bin/bash

cd /workdir/NPB/NPB3.4-OMP/
python3 generate_sampling_points.py --output_dir=/workdir/experiments/prefetcher/gem5_configurations/npb_sampling_points/ --workload cg.E --sampling_site=1
python3 generate_sampling_points.py --output_dir=/workdir/experiments/prefetcher/gem5_configurations/npb_sampling_points/ --workload cg.E --sampling_site=2
python3 generate_sampling_points.py --output_dir=/workdir/experiments/prefetcher/gem5_configurations/npb_sampling_points/ --workload is.D --sampling_site=1
