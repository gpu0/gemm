#!/bin/bash
rm -rf mat.txt
/opt/rocm/hip/bin/hipcc --amdgpu-target=gfx803 --amdgpu-target=gfx900 gemm128x128v2.cpp
