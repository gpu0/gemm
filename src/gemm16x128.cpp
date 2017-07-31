#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define TID_X 256

#define A_X 256
#define A_Y 16
#define B_Y A_X
#define B_X A_Y
#define C_X B_X
#define C_Y A_Y

template<typename T, typename T2>
__global__ void Gemm128x128(T *A, T *B, T* C) {
    int tx = hipThreadIdx_x;

    __shared__ T sA[128*8];
    __shared__ T sB[128*8];

    int gmLoadBtx  = tx%8 + (tx/8)*16;
    int ldsStoreBtx = (tx%8)*128 + tx/8;

    for(int i=0;i<(128*8)/256;i++) {
        sB[ldsStoreBtx] = B[gmLoadBtx];
    }

}


int main() {
}
