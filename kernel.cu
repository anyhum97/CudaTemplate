#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Reflection.cu"

void __global__ CudaSample(float* buf)
{
    /// <<<1, 128>>>

    const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block > 0 || thread > 128)
    {
        return;
    }

    buf[thread] = thread;
}

int main()
{
    cudaSetDevice(0);

    Reflection<float> buffer(128);

    CudaSample<<<1, 128>>>(Device(buffer));

    buffer.Receive();



    return 0;
}