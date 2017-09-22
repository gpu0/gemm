#include<iostream>
#include<hip/hip_runtime_api.h>
#include<hip/hip_runtime.h>

#define LEN 64
#define SIZE (LEN * sizeof(float))

__global__ void ForLoopCode(float *Out, float *In) {
    int tx = hipThreadIdx_x;
    float in = In[tx];
    float out = Out[tx];
    asm volatile("\n \
    s_movk_i32 s0, 0 \n \
BB0_4:\n\
    v_add_f32 %0, %1, %2 \n \
    s_addk_i32 s0, 0x1 \n \
    s_cmp_lt_u32 s0, 0x10 \n \
    s_cbranch_scc1 BB0_4 \n \
    ": "=v"(out): "v"(out), "v"(in));
    Out[tx] = out;
}

int main() {
    float *dDst, *dSrc, *hSrc, *hDst;
    hSrc = new float[LEN];
    hDst = new float[LEN];
    for(int i=0;i<LEN;i++) {
        hSrc[i] = 1.0f;
        hDst[i] = 1.0f;
    }
    hipMalloc(&dSrc, SIZE);
    hipMalloc(&dDst, SIZE);
    hipMemcpy(dDst, hDst, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(dSrc, hSrc, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(ForLoopCode, dim3(1,1,1), dim3(LEN,1,1), 0, 0, dDst, dSrc);
    hipDeviceSynchronize();
    hipMemcpy(hDst, dDst, SIZE, hipMemcpyDeviceToHost);
    std::cout<<hDst[10]<<std::endl;
}
