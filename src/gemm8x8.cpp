#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define TID_X 256

#define A_X 128
#define A_Y 8
#define B_Y A_X
#define B_X A_Y
#define C_X B_X
#define C_Y A_Y

template<typename T, typename T2, typename T4>
__global__ void Gemm8x8(T *A, T *B, T* C) {
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
        sA[ldsStoreAtx + i*256] = A[gmLoadAtx + i*256];
    }

    T rA[8*8];
    T rB[8*8];
    T rC[8*8];

    for(int i=0;i<8*8;i++) {
        rA[i] = sA[i+tx*64];
        rB[i] = sB[i+tx*64];
        rC[i] = 0;
    }

    for(int j=0;j<8;j++) {
        for(int i=0;i<8;i++) {
            int idC = i+j*8;
            rC[idC] = rA[i*8+0] * rB[j*8+0];
            rC[idC] += rA[i*8+1] * rB[j*8+1];
            rC[idC] += rA[i*8+2] * rB[j*8+2];
            rC[idC] += rA[i*8+3] * rB[j*8+3];

            rC[idC] += rA[i*8+4] * rB[j*8+4];
            rC[idC] += rA[i*8+5] * rB[j*8+5];
            rC[idC] += rA[i*8+6] * rB[j*8+6];
            rC[idC] += rA[i*8+7] * rB[j*8+7];
        }
    }

    if(tx == 0){
        for(int j=0;j<8;j++) {
            for(int i=0;i<8;i++) {
                C[i+j*8] = rC[i+j*8];
            }
        }
    }

}


int main() {

    hipSetDevice(1);

    std::vector<float> A(A_X*A_Y);
    std::vector<float> B(B_X*B_Y);
    std::vector<float> C(C_X*C_Y);

    float *Ad, *Bd, *Cd;
    hipMalloc(&Ad, sizeof(float)*A_X*A_Y);
    hipMalloc(&Bd, sizeof(float)*B_X*B_Y);
    hipMalloc(&Cd, sizeof(float)*C_X*C_Y);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), 1.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    hipMemcpy(Ad, A.data(), sizeof(float)*A_X*A_Y, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B.data(), sizeof(float)*B_X*B_Y, hipMemcpyHostToDevice);
    hipMemcpy(Cd, C.data(), sizeof(float)*C_X*C_Y, hipMemcpyHostToDevice);

    hipLaunchKernelGGL((Gemm8x8<float, float2, float4>), dim3(1,1,1), dim3(16,16,1), 0, 0, Ad, Bd, Cd);
    hipDeviceSynchronize();

    hipMemcpy(C.data(), Cd, sizeof(float)*C_X*C_Y, hipMemcpyDeviceToHost);

    std::cout<<C[0]<<std::endl;

}
