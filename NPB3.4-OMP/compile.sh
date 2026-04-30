#!/bin/bash

rm -rf bin/
mkdir bin

WORKLOADS=("IS" "CG")
CLASSES=("S" "W" "A" "B" "C")

for workload in "${WORKLOADS[@]}"
do
    for workload_class in "${CLASSES[@]}"
    do
        ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make ${workload} CLASS=${workload_class}
    done

done
