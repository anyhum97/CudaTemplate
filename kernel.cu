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

    buf[thread] += thread;
}

Reflection<float> buffer;

float values[128] = { -0.9f, 2.2f, 3.5f };

void CudaInit()
{
    cudaSetDevice(0);

    buffer = Malloc<float>(values, 128, true);
}

void CudaFree()
{
    Free(buffer);
}

int main()
{
    CudaInit();

    ////////////////////////////////////////////////////////////////////////

    CudaSample<<<1, 128>>>(Device(buffer));

    Receive(buffer, 128);

    Show(buffer);

    ////////////////////////////////////////////////////////////////////////

    CudaFree();
    return 0;
}