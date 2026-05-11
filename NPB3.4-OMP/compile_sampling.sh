#!/bin/bash

rm -rf bin/
mkdir bin

WORKLOADS=("IS_SAMPLING" "CG_SAMPLING")
CLASSES=("S" "D" "E")

for workload in "${WORKLOADS[@]}"
do
    for workload_class in "${CLASSES[@]}"
    do
        ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make ${workload} CLASS=${workload_class}
    done

done
