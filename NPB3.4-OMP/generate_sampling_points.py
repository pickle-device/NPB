# Copyright (c) 2026 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import gzip
import os
import random

class SamplingPoint:
    def __init__(self, starting_iter, num_warmup_iters):
        self.starting_iter = starting_iter
        self.num_warmup_iters = num_warmup_iters

# Format:
# """
# NAS Parallel Benchmarks (NPB3.4-OMP) - IS Benchmark
# 
# Size:  2147483648  (class D)
# Iterations:  10
# Number of available threads:  8
# 
# i=0, m=0
# i=1, m=1
# i=2, m=2
# i=3, m=3
# i=5, m=14
# """
# where i is the bucket, and m is the starting index of that bucket.
# For class D, the index and the data are 8 bytes long.
# Note:
#   - We print all the traces of all rank() calls with unmodified NPB code.
#   - We only benchmark rank(2) call in our version.
#   - The order of the call in the unmodified NPB code is: rank(1) (for warmup), rank(1), rank(2), rank(3), etc.
#   - So, we'll collect the third trace

def generate_sampling_points_for_is(
    workload_name, workload_class, llc_size_bytes, num_sampling_points
):
    # read array structure from trace file
    trace_file = f"{workload_name}.{workload_class}.array_data.gz"
    if not os.path.exists(trace_file):
        print(f"Error: Trace file {trace_file} not found.")
        exit(1)
    with gzip.open(trace_file, "rt") as f:
        lines = f.readlines()
    bucket_start_indices = {}
    # skip to the line that starts with "i="
    # stop once we read at least 1 line with "i=" and the next line does not start with "i="
    reading_buckets = False
    rank_call_count = 1
    for line in lines:
        if reading_buckets and not line.startswith("i="):
            reading_buckets = False
            rank_call_count += 1
            if rank_call_count > 3:
                break
        if line.startswith("i="):
            reading_buckets = True
            if rank_call_count == 3:
                parts = line.split(", ")
                bucket = int(parts[0].split("=")[1].strip())
                start_index = int(parts[1].split("=")[1].strip())
                bucket_start_indices[bucket] = start_index
    bucket_to_num_elements = {}
    num_buckets = len(bucket_start_indices)
    for bucket in range(1, num_buckets):
        bucket_to_num_elements[bucket] = bucket_start_indices[bucket] - bucket_start_indices[bucket - 1]
    # now we have the bucket structure, we can generate random sampling points
    # For IS,
    # - starting_iter is the starting bucket
    # - num_warmup_iters is the number of buckets to fill the LLC before sampling
    # - num_elements_to_sample is the number of elements (not the number of buckets) to sample after the warmup iterations
    sampling_points = []
    while len(sampling_points) < num_sampling_points:
        starting_bucket = random.randint(1, num_buckets - 1)
        num_warmup_buckets = 0
        remaining_llc_size = llc_size_bytes
        successfully_generated = False
        while remaining_llc_size > 0 and starting_bucket + num_warmup_buckets < num_buckets:
            num_elements = bucket_to_num_elements[starting_bucket + num_warmup_buckets]
            num_warmup_buckets += 1
            # for each element, we pull in 8 bytes from the indexing array and 8 bytes from the data array
            remaining_llc_size -= num_elements * 16
        successfully_generated = starting_bucket + num_warmup_buckets < num_buckets
        if successfully_generated:
            sampling_points.append(SamplingPoint(starting_bucket, num_warmup_buckets))
    print(f"Generated {len(sampling_points)} sampling points:")
    for i, sp in enumerate(sampling_points):
        print(f"  Sampling Point {i + 1}: Starting Bucket = {sp.starting_iter}, Warmup Buckets = {sp.num_warmup_iters}")
        
if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser(
        "Randomly generate sampling points for NPB workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["cg.E", "is.D"],
        required=True,
        help="NPB workload to generate sampling points for",
    )
    parser.add_argument(
        "--llc_size_mib",
        type=int,
        default=32,
        help="LLC size in MiB",
    )
    parser.add_argument(
        "--num_sampling_points",
        type=int,
        default=30,
        help="Number of sampling points to generate",
    )

    args = parser.parse_args()
    workload_name, workload_class = args.workload.split(".")
    llc_size_bytes = args.llc_size_mib * 1024 * 1024
    num_sampling_points = args.num_sampling_points

    print(f"Input:")
    print(f"  Workload: {workload_name} Class {workload_class}")
    print(f"  LLC Size: {args.llc_size_mib} MiB")
    print(f"  Number of Sampling Points: {num_sampling_points}")

    # read array structure from array data trace
    if workload_name == "is":
        generate_sampling_points_for_is(workload_name, workload_class, llc_size_bytes, num_sampling_points)
