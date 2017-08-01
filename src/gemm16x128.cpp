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

template<typename T, typename T2, typename T4>
__global__ void Gemm128x128(T *A, T *B, T* C) {
    int tx = hipThreadIdx_x;

    __shared__ T sA[128*8];
    __shared__ T sB[128*8];

    int gmLoadBtx  = tx%8 + (tx/8)*16;
    int ldsStoreBtx = (tx%8)*128 + tx/8;

    for(int i=0;i<(128*8)/256;i++) {
        sB[ldsStoreBtx + i*256 ] = B[gmLoadBtx + i*256];
    }

    int gmLoadAtx = tx;
    int ldsStoreAtx = tx;

    for(int i=0;i<(128*8)/256;i++) {
        sA[gmLoadAtx + i*256] = A[gmLoadAtx + i*256];
    }

    T rA[8*8];
    T rB[8*8];
    T rC[8*8];

    for(int i=0;i<8*8;i++) {
        rA[i] = sA[i+tx*64];
        rB[i] = sB[i+tx*64];
    }

}


int main() {
}
