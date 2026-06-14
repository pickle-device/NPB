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

class UA_SamplingPoint:
    def __init__(self, ua_step, cg_iteration, starting_element, num_warmup_elements):
        self.ua_step = ua_step
        self.cg_iteration = cg_iteration
        self.starting_element = starting_element
        self.num_warmup_elements = num_warmup_elements

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
    return sampling_points

# Format:
# """
# NAS Parallel Benchmarks (NPB3.4-OMP) - CG Benchmark                                                
#                                                                                                    
# Size:     9000000                                                                                  
# Iterations:                    100                                                                 
# Number of available threads:    64                                                                 
#                                                                                                    
#     9000000                                                                                        
# Number of non-zeros in A:           6326754836                                                     
# row,starts at index                                                                                
#           1                    1                                                                   
#           2                  626                                                                   
#           3                 1277                                                                   
#           4                 1954                                                                   
#           5                 2787                                                                   
#           6                 3516  
# """
# Note:
# - This is the CSR's row pointer array of the A matrix. This is 1-indexed.
# - We can use this to determine how many non-zeros are in each row, and thus how much data is accessed in each iteration of the CG kernel.
# - Even though the CG kernel has two sampling sites, the array access pattern is the same for both sampling sites, so we can use the same array structure to generate sampling points for both sampling sites.
def generate_sampling_points_for_cg(
    workload_name, workload_class, llc_size_bytes, num_sampling_points
):
    # read array structure from trace file
    trace_file = f"{workload_name}.{workload_class}.array_data.gz"
    if not os.path.exists(trace_file):
        print(f"Error: Trace file {trace_file} not found.")
        exit(1)
    with gzip.open(trace_file, "rt") as f:
        lines = f.readlines()
    reading_rows = False
    row_start_indices = [0]
    for line in lines:
        line = line.strip()
        if not reading_rows:
            if line.startswith("row,starts at index"):
                reading_rows = True
            continue
        if reading_rows:
            line = line.strip()
            parts = line.split()
            row_start_indices.append(int(parts[1]))
    row_to_num_nonzeros = {}
    num_rows = len(row_start_indices)
    for row in range(1, num_rows):
        num_nonzeros = row_start_indices[row] - row_start_indices[row-1]
        row_to_num_nonzeros[row] = num_nonzeros
    # now we have the bucket structure, we can generate random sampling points
    # For CG,
    # - starting_iter is the starting row
    # - num_warmup_iters is the number of rows to fill the LLC before sampling
    sampling_points = []
    while len(sampling_points) < num_sampling_points:
        starting_row = random.randint(1, num_rows - 1)
        num_warmup_rows = 0
        remaining_llc_size = llc_size_bytes
        successfully_generated = False
        while remaining_llc_size > 0 and starting_row + num_warmup_rows < num_rows:
            num_nonzeros = row_to_num_nonzeros[starting_row + num_warmup_rows]
            num_warmup_rows += 1
            # for each element, we pull in 4 bytes from the column index array and 8 bytes from the data array
            remaining_llc_size -= (4 + 8) * num_nonzeros
            successfully_generated = starting_row + num_warmup_rows < num_rows
        if successfully_generated:
            sampling_points.append(SamplingPoint(starting_row, num_warmup_rows))
    print(f"Generated {len(sampling_points)} sampling points:")
    for i, sp in enumerate(sampling_points):
        print(f"  Sampling Point {i + 1}: Starting Row = {sp.starting_iter}, Warmup Rows = {sp.num_warmup_iters}")
    return sampling_points

# For UA benchmark class D,
# - There are 250 UA steps.
# - In each UA step, there are 10 CG iterations.
# - In each CG iteration, transf and transfb will be called once. These are the functions we want to profile.
# - We will pick the transf/transfb call at a certain CG iteration of a certain UA step. This is one sampling point.
# - lx1=5, lnje=2, nsides=6 (or 6 faces), num elements=514400, lmor=19134400
# - For each element that transf/transfb process, the function pulls in,
#   - lx1 * lx1 * nsides for idel, each is 4 bytes -> 600 bytes total
#   - each idel indexes one tx, each tx is 8 bytes -> 5 * 5 * 6 * 8 = 1200 bytes total
#   - between 29 and 100 (inclusive) for each side (or each face) for idmo, each is 4 bytes -> at least 29 * 6 * 4 = 696 bytes total
#   - each idmo indexes one pmorx, each pmorx is 8 bytes -> at least 29 * 6 * 8 = 1392 bytes total
# - In total, for each element that transf/transfb process, the function pulls in, at least (600 + 1200 + 696 + 1392) = 3888 bytes.
# - So to warm up the cache, we need ceil(llc_size / 3888) elements to be processed by transf/transfb.
# - Though, note that, the average number of idmo access is way larger than 29.
def generate_sampling_points_for_ua_class_D(
    workload_name, workload_class, llc_size_bytes, num_sampling_points
):
    sampling_points = []
    while len(sampling_points) < num_sampling_points:
        ua_step = random.randint(1, 250) # skip the step 0, which should not be profiled
        cg_iteration = random.randint(1, 10) # 1 to 10, inclusive
        starting_element = 10 # we want to evaluate the whole function call
        num_warmup_elements = llc_size_bytes // 3888 + 1
        successfully_generated = (starting_element + num_warmup_elements + 10000) < 514400
        if successfully_generated:
            sampling_points.append(UA_SamplingPoint(ua_step, cg_iteration, starting_element, num_warmup_elements))
    print(f"Generated {len(sampling_points)} sampling points:")
    for i, sp in enumerate(sampling_points):
        print(f"  Sampling Point {i + 1}: UA Step = {sp.ua_step}, CG Iteration = {sp.cg_iteration}, Starting Element = {sp.starting_element}, Warmup Elements = {sp.num_warmup_elements}")
    return sampling_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Randomly generate sampling points for NPB workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output folder for storing the file with the generated sampling points",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices={"cg.E", "is.D", "ua.D"},
        required=True,
        help="NPB workload to generate sampling points for",
    )
    parser.add_argument(
        "--sampling_site",
        type=int,
        required=True,
        help="Sampling site",
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
    sampling_site = args.sampling_site
    llc_size_bytes = args.llc_size_mib * 1024 * 1024
    num_sampling_points = args.num_sampling_points
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    random.seed(42 + sampling_site)

    if workload_name in {"is"}:
        assert sampling_site in {1}, f"{workload_name} only has one sampling site"
    elif workload_name in {"cg"}:
        assert sampling_site in {1, 2}, f"{workload_name} has two sampling sites"
    elif workload_name in {"ua"}:
        assert sampling_site in {1, 2, 3, 4}, f"{workload_name} has four sampling sites"
        assert workload_class in {"D"}, f"{workload_name}.{workload_class} not implemented yet"
    else:
        raise ValueError(f"Unknown workload {workload_name}")

    output_file_name = f"{workload_name}.{workload_class}.sampling_site-{sampling_site}.llc-{args.llc_size_mib}MiB.sampling_points.txt"
    output_file_path = os.path.join(output_dir, output_file_name)

    print(f"Input:")
    print(f"  Workload: {workload_name}")
    print(f"  Class: {workload_class}")
    print(f"  Sampling Site: {sampling_site}")
    print(f"  LLC Size: {args.llc_size_mib} MiB")
    print(f"  Number of Sampling Points: {num_sampling_points}")

    # read array structure from array data trace
    if workload_name == "is":
        sampling_points = generate_sampling_points_for_is(workload_name, workload_class, llc_size_bytes, num_sampling_points)
    elif workload_name == "cg":
        sampling_points = generate_sampling_points_for_cg(workload_name, workload_class, llc_size_bytes, num_sampling_points)
    elif workload_name == "ua":
        sampling_points = generate_sampling_points_for_ua_class_D(workload_name, workload_class, llc_size_bytes, num_sampling_points)
    else:
        raise ValueError(f"Unknown workload {workload_name}")

    # write sampling points to output file
    with open(output_file_path, "w") as f:
        f.write(f"Input:")
        f.write(f"  Workload: {workload_name}\n")
        f.write(f"  Class: {workload_class}\n")
        f.write(f"  Sampling Site: {sampling_site}\n")
        f.write(f"  LLC Size: {args.llc_size_mib} MiB\n")
        f.write(f"  Number of Sampling Points: {num_sampling_points}\n")
        f.write(f"  Sampling Points:\n")
        if workload_name in {"is", "cg"}:
            for i, sp in enumerate(sampling_points):
                assert isinstance(sp, SamplingPoint)
                f.write(f"    Sampling Point {i + 1}: Starting Iter = {sp.starting_iter}, Num Warmup Iters = {sp.num_warmup_iters}\n")
        elif workload_name in {"ua"}:
            for i, sp in enumerate(sampling_points):
                assert isinstance(sp, UA_SamplingPoint)
                f.write(f"    Sampling Point {i + 1}: UA Step = {sp.ua_step}, CG Iter = {sp.cg_iteration}, Starting Element = {sp.starting_element}, Num Warmup Elements = {sp.num_warmup_elements}\n")
