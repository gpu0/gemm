#include<iostream>
#include<fstream>
#include<string>
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

__device__ inline void outerProd(float4 &a0, float4 &a1, float4 &b0, float4 &b1, float4 c[]) {
    c[0].x += a0.x * b0.x;
    c[0].y += a0.x * b0.y;
    c[0].z += a0.x * b0.z;
    c[0].w += a0.x * b0.w;

    c[1].x += a1.x * b0.x;
    c[1].y += a1.x * b0.y;
    c[1].z += a1.x * b0.z;
    c[1].w += a1.x * b0.w;

    c[2].x += a0.y * b0.x;
    c[2].y += a0.y * b0.y;
    c[2].z += a0.y * b0.z;
    c[2].w += a0.y * b0.w;

    c[3].x += a1.y * b0.x;
    c[3].y += a1.y * b0.y;
    c[3].z += a1.y * b0.z;
    c[3].w += a1.y * b0.w;

    c[4].x += a0.z * b0.x;
    c[4].y += a0.z * b0.y;
    c[4].z += a0.z * b0.z;
    c[4].w += a0.z * b0.w;

    c[5].x += a1.z * b0.x;
    c[5].y += a1.z * b0.y;
    c[5].z += a1.z * b0.z;
    c[5].w += a1.z * b0.w;

    c[6].x += a0.w * b0.x;
    c[6].y += a0.w * b0.y;
    c[6].z += a0.w * b0.z;
    c[6].w += a0.w * b0.w;

    c[7].x += a1.w * b0.x;
    c[7].y += a1.w * b0.y;
    c[7].z += a1.w * b0.z;
    c[7].w += a1.w * b0.w;

    c[8].x += a0.x * b1.x;
    c[8].y += a0.x * b1.y;
    c[8].z += a0.x * b1.z;
    c[8].w += a0.x * b1.w;

    c[9].x += a1.x * b1.x;
    c[9].y += a1.x * b1.y;
    c[9].z += a1.x * b1.z;
    c[9].w += a1.x * b1.w;

    c[10].x += a0.y * b1.x;
    c[10].y += a0.y * b1.y;
    c[10].z += a0.y * b1.z;
    c[10].w += a0.y * b1.w;

    c[11].x += a1.y * b1.x;
    c[11].y += a1.y * b1.y;
    c[11].z += a1.y * b1.z;
    c[11].w += a1.y * b1.w;

    c[12].x += a0.z * b1.x;
    c[12].y += a0.z * b1.y;
    c[12].z += a0.z * b1.z;
    c[12].w += a0.z * b1.w;

    c[13].x += a1.z * b1.x;
    c[13].y += a1.z * b1.y;
    c[13].z += a1.z * b1.z;
    c[13].w += a1.z * b1.w;

    c[14].x += a0.w * b1.x;
    c[14].y += a0.w * b1.y;
    c[14].z += a0.w * b1.z;
    c[14].w += a0.w * b1.w;

    c[15].x += a1.w * b1.x;
    c[15].y += a1.w * b1.y;
    c[15].z += a1.w * b1.z;
    c[15].w += a1.w * b1.w;
}

__device__ inline void loadCtoRegs(float4* C, float4 c[], int gmStoreCtx) {
    c[0]  = C[gmStoreCtx + 0*32 + 0];
    c[1]  = C[gmStoreCtx + 0*32 + 1];
    c[2]  = C[gmStoreCtx + 1*32 + 0];
    c[3]  = C[gmStoreCtx + 1*32 + 1];
    c[4]  = C[gmStoreCtx + 2*32 + 0];
    c[5]  = C[gmStoreCtx + 2*32 + 1];
    c[6]  = C[gmStoreCtx + 3*32 + 0];
    c[7]  = C[gmStoreCtx + 3*32 + 1];
    c[8]  = C[gmStoreCtx + 4*32 + 0];
    c[9]  = C[gmStoreCtx + 4*32 + 1];
    c[10] = C[gmStoreCtx + 5*32 + 0];
    c[11] = C[gmStoreCtx + 5*32 + 1];
    c[12] = C[gmStoreCtx + 6*32 + 0];
    c[13] = C[gmStoreCtx + 6*32 + 1];
    c[14] = C[gmStoreCtx + 7*32 + 0];
    c[15] = C[gmStoreCtx + 7*32 + 1];
}

__global__ void Gemm128x128(float4 *A, float4 *B, float4 *C) {
    int tx = hipThreadIdx_x;

/*
    __shared__ float4 sAx4[128*2];
    __shared__ float4 sBx4[128*2];

    int gmLoadAtx = tx;
    int gmLoadBtx = tx;

    int ldsStoreAtx = tx;
    int ldsStoreBtx = tx;

    int gmStoreCtx = tx;

    float4 a0, a1, b0, b1, c[8*2];

    loadCtoRegs(C, c, gmStoreCtx);

    sAx4[ldsStoreAtx] = A[gmLoadAtx];
    sBx4[ldsStoreBtx] = B[gmLoadBtx];

    a0 = sAx4[tx];
    a1 = a0;

    b0 = sBx4[tx];
    b1 = b0;

    outerProd(a0, a1, b0, b1, c);
*/
    #define STRIDE 32
    int gmStoreCtx = (tx%16)*2 + (tx/16)*16*16;
    float4 a0 = {1, 1, 1, 1};

    C[gmStoreCtx + 0*STRIDE + 0] = a0;//c[0];
    C[gmStoreCtx + 0*STRIDE + 1] = a0;

    C[gmStoreCtx + 1*STRIDE + 0] = a0;
    C[gmStoreCtx + 1*STRIDE + 1] = a0;



}

int main(){
    hipSetDevice(1);
    std::vector<float> A(A_X*A_Y);
    std::vector<float> B(B_X*B_Y);
    std::vector<float> C(C_X*C_Y);

    std::fill(A.begin(), A.end(), 2.0f);
    std::fill(B.begin(), B.end(), 2.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    float *Ad, *Bd, *Cd;
    hipMalloc(&Ad, A.size()*sizeof(float));
    hipMalloc(&Bd, B.size()*sizeof(float));
    hipMalloc(&Cd, C.size()*sizeof(float));

    hipMemcpy(Ad, A.data(), A.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Bd, B.data(), B.size()*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(Cd, C.data(), C.size()*sizeof(float), hipMemcpyHostToDevice);

    std::cout<<"Size of C Matrix: "<<C_X<<"x"<<C_Y<<" = "<<C_X*C_Y<<std::endl;

    std::cout<<"Range of Ad is: "<<Ad<<" "<<Ad+A.size()<<std::endl;
    std::cout<<"Range of Bd is: "<<Bd<<" "<<Bd+B.size()<<std::endl;
    std::cout<<"Range of Cd is: "<<Cd<<" "<<Cd+C.size()<<std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL((Gemm128x128), dim3(1,1,1), dim3(TID_X,TID_Y,1), 0, 0, (float4*)Ad, (float4*)Bd, (float4*)Cd);
    hipDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
    double perf = (double)(C_X*C_Y*C_Y)*2/1.0E12/elapsedSec;
    std::cout<<"TFLOPs: "<<perf<<std::endl;
    std::cout<<"Projected TFLOPs: "<<perf*64<<std::endl;

    hipMemcpy(C.data(), Cd, C.size()*sizeof(float), hipMemcpyDeviceToHost);

    std::fstream fs;
    fs.open("mat.txt", std::fstream::in | std::fstream::out | std::fstream::app);
    for(int j=0;j<C_Y;j++) {
        fs << "j = "<<std::to_string(j)<<"\n";
        for(int i=0;i<C_X;i++) {
            fs << std::to_string(C[i+j*C_X]) <<" ";
        }
        fs <<"\n";
    }

    for(int j=0;j<C_Y;j++) {
        for(int i=0;i<C_X;i++) {
            if(C[i+j*C_X] != 2) { std::cerr<<"Bad output: "<<C[i+j*C_X]<<" at: "<<i<<"x"<<j<<" = "<<i+j*C_X<<std::endl; return 0;}
        }
    }

}
