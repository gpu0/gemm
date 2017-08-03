#!/bin/bash

/opt/rocm/hip/bin/hipcc --amdgpu-target=gfx803 --amdgpu-target=gfx900 gemm128x128.cpp
