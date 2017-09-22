#include<iostream>
#include<hip/hip_runtime_api.h>
#include<hip/hip_runtime.h>

#define LEN 64
#define SIZE (LEN * sizeof(float))

__global__ void ForLoopCode(float *Out, float *In) {
    int tx = hipThreadIdx_x;
    float in = In[tx];
    float out = Out[tx];
    for(int i=0;i<1024;i++) {
        out += in;
    }
    Out[tx] = out;
}

int main() {
    float *dDst, *dSrc;
    hipLaunchKernelGGL(ForLoopCode, dim3(1,1,1), dim3(LEN,1,1), 0, 0, dDst, dSrc);
    
}
