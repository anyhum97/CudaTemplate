#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Reflection.cu"

////////////////////////////////////////////////////////////////////////

const unsigned int Width = 4;
const unsigned int Height = 4;

////////////////////////////////////////////////////////////////////////

Reflection<float> Buffer;   // [3][Width][Height];

////////////////////////////////////////////////////////////////////////

__inline__ __device__ unsigned int GetBufferIndex(const unsigned int dim, int x, int y)
{
    // Buffer[3][Width][Height];

    return dim*Width*Height + x*Height + y;
}

////////////////////////////////////////////////////////////////////////

void __global__ BufferAccess(float* Buffer)
{
    /// <<<Width, Height>>>

    const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block >= Width || thread >= Height)
    {
        return;
    }

    Buffer[GetBufferIndex(0, block, thread)] = 1.0f;
    Buffer[GetBufferIndex(1, block, thread)] = 2.0f;
    Buffer[GetBufferIndex(2, block, thread)] = 3.0f;
}

////////////////////////////////////////////////////////////////////////

cudaEvent_t start;
cudaEvent_t stop;

////////////////////////////////////////////////////////////////////////

void CudaMalloc()
{
    cudaSetDevice(0);

    Buffer = Malloc<float>(3*Width*Height);
}

void CudaFree()
{
    Free(Buffer);
}

////////////////////////////////////////////////////////////////////////

void Test()
{
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

    ////////////////////////////////////////////////////////////////////////

    BufferAccess<<<Width, Height>>>(Device(Buffer));

    ////////////////////////////////////////////////////////////////////////

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 0;

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    ////////////////////////////////////////////////////////////////////////

    cout << time << "ms [OK]\n\n";

    ////////////////////////////////////////////////////////////////////////
}

void main()
{
    CudaMalloc();

    ////////////////////////////////////////////////////////////////////////

    Test();
    Receive(Buffer);
    Show(Buffer);

    ////////////////////////////////////////////////////////////////////////

    CudaFree();
}