#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>


#define TID_X 256
#define TID_Y 1

#define A_X 128
#define A_Y 128
#define B_X 128
#define B_Y A_X
#define C_X B_X
#define C_Y A_X


__global__ void Gemm128x128(float4 *A, float4 *B, float4 *C) {
    int tx = hipThreadIdx_x;
    __shared__ float4 sAx4[128*2];
    __shared__ float4 sBx4[128*2];

    int gmLoadAtx = tx;
    int gmLoadBtx = tx;

    inti ldsStoreAtx = (tx%2) * 128 * 4 + tx;
    int ldsStoreBtx = tx;

    float *sA = (float*)sAx4;

    float4 a = A[gmLoadAtx];

    sA[sAtx + 0 * 128] = a.x;
    sA[sAtx + 1 * 128] = a.y;
    sA[sAtx + 2 * 128] = a.z;
    sA[sAtx + 3 * 128] = a.w;

    sBx4[ldsStoreBtx] = B[gmLoadBtx];

    float4 rA0x4, rA1x4, rB0x4, rB1x4, rC[8][2];
    rA0x4 = sAx4[tx%16];
    rA1x4 = sAx4[tx%16+16];
    rB0x4 = sBx4[tx%16];
    rB1x4 = sBx4[tx%16+16];

    rC[0][0].x = rA0x4.x * rB0x4.x;
    

}

int main(){}
