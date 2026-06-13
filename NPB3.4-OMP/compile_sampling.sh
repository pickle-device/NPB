#!/bin/bash

rm -rf bin/*.x.sampling.*
mkdir bin

WORKLOADS=("IS_SAMPLING" "CG_SAMPLING" "UA_SAMPLING")
CLASSES=("S" "D" "E")

# IS_SAMPLING
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make is_sampling CLASS=S -j $(nproc)
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make is_sampling CLASS=D -j $(nproc)
# CG_SAMPLING
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make cg_sampling CLASS=S -j $(nproc)
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make cg_sampling CLASS=E -j $(nproc)
# UA_SAMPLING
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make ua_sampling CLASS=S -j $(nproc)
ENABLE_GEM5=1 ENABLE_PICKLEDEVICE=1 make ua_sampling CLASS=D -j $(nproc)
