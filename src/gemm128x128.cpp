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

    int sAtx = (tx%2) * 128 * 4 + tx/2;
    int ldsStoreBtx = tx;

    float *sA = (float*)sAx4;
    float4 a = A[gmLoadAtx];

    sA[sAtx + 0 * 128] = a.x;
    sA[sAtx + 1 * 128] = a.y;
    sA[sAtx + 2 * 128] = a.z;
    sA[sAtx + 3 * 128] = a.w;

    sBx4[ldsStoreBtx] = B[gmLoadBtx];

    float4 a0, a1, b0, b1, c[8][2];
    a0 = sAx4[tx%16];
    a1 = sAx4[tx%16+16];

    b0 = sBx4[tx%16];
    b1 = sBx4[tx%16+16];

  c[0][0].x = a0.x * b0.x;
  c[0][0].y = a0.x * b0.y;
  c[0][0].z = a0.x * b0.z;
  c[0][0].w = a0.x * b0.w;

  c[0][1].x = a1.x * b0.x;
  c[0][1].y = a1.x * b0.y;
  c[0][1].z = a1.x * b0.z;
  c[0][1].w = a1.x * b0.w;

  c[1][0].x = a0.y * b0.x;
  c[1][0].y = a0.y * b0.y;
  c[1][0].z = a0.y * b0.z;
  c[1][0].w = a0.y * b0.w;

  c[1][1].x = a1.y * b0.x;
  c[1][1].y = a1.y * b0.y;
  c[1][1].z = a1.y * b0.z;
  c[1][1].w = a1.y * b0.w;

  c[2][0].x = a0.z * b0.x;
  c[2][0].y = a0.z * b0.y;
  c[2][0].z = a0.z * b0.z;
  c[2][0].w = a0.z * b0.w;

  c[2][1].x = a1.z * b0.x;
  c[2][1].y = a1.z * b0.y;
  c[2][1].z = a1.z * b0.z;
  c[2][1].w = a1.z * b0.w;

  c[3][0].x = a0.w * b0.x;
  c[3][0].y = a0.w * b0.y;
  c[3][0].z = a0.w * b0.z;
  c[3][0].w = a0.w * b0.w;

  c[3][1].x = a1.w * b0.x;
  c[3][1].y = a1.w * b0.y;
  c[3][1].z = a1.w * b0.z;
  c[3][1].w = a1.w * b0.w;

  c[4][0].x = a0.x * b1.x;
  c[4][0].y = a0.x * b1.y;
  c[4][0].z = a0.x * b1.z;
  c[4][0].w = a0.x * b1.w;

  c[4][1].x = a1.x * b1.x;
  c[4][1].y = a1.x * b1.y;
  c[4][1].z = a1.x * b1.z;
  c[4][1].w = a1.x * b1.w;

  c[5][0].x = a0.y * b1.x;
  c[5][0].y = a0.y * b1.y;
  c[5][0].z = a0.y * b1.z;
  c[5][0].w = a0.y * b1.w;

  c[5][1].x = a1.y * b1.x;
  c[5][1].y = a1.y * b1.y;
  c[5][1].z = a1.y * b1.z;
  c[5][1].w = a1.y * b1.w;

  c[6][0].x = a0.z * b1.x;
  c[6][0].y = a0.z * b1.y;
  c[6][0].z = a0.z * b1.z;
  c[6][0].w = a0.z * b1.w;

  c[6][1].x = a1.z * b1.x;
  c[6][1].y = a1.z * b1.y;
  c[6][1].z = a1.z * b1.z;
  c[6][1].w = a1.z * b1.w;

  c[7][0].x = a0.w * b1.x;
  c[7][0].y = a0.w * b1.y;
  c[7][0].z = a0.w * b1.z;
  c[7][0].w = a0.w * b1.w;

    c[7][1].x = a1.w * b1.x;
    c[7][1].y = a1.w * b1.y;
    c[7][1].z = a1.w * b1.z;
    c[7][1].w = a1.w * b1.w;

    int gmStoreCtx = (tx%16)*2 + (tx/16)*8*128;
    C[gmStoreCtx+0]    = c[0][0];
    C[gmStoreCtx+1]    = c[0][1];
    C[gmStoreCtx+32+0] = c[1][0];
    C[gmStoreCtx+32+1] = c[1][1];
}

int main(){
    std::vector<float> A(A_X*A_Y);
    std::vector<float> B(B_X*B_Y);
    std::vector<float> C(C_X*C_Y);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 1.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    float *Ad, *Bd, *Cd;
    hipMalloc(&Ad, A.size()*sizeof(float));
    hipMalloc(&Bd, B.size()*sizeof(float));
    hipMalloc(&Cd, C.size()*sizeof(float));

    hipMemcpy(Ad, A.data(), A.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Bd, B.data(), B.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Cd, C.data(), C.size()*sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL((Gemm128x128), dim3(1,1,1), dim3(16*16,1), 0, 0, (float4*)Ad, (float4*)Bd, (float4*)Cd);
    hipDeviceSynchronize();

    hipMemcpy(C.data(), Cd, C.size()*sizeof(float), hipMemcpyDeviceToHost);

    std::cout<<C[10]<<std::endl;

}
