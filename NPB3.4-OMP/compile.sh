#!/bin/bash

rm -rf bin/*.x.m5.pdev
mkdir bin

WORKLOADS=("IS" "CG" "UA")
CLASSES=("S" "W" "A" "B" "C")

for workload in "${WORKLOADS[@]}"
do
    for workload_class in "${CLASSES[@]}"
    do
        ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make ${workload} CLASS=${workload_class}
    done

done
