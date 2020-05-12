#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Reflection.cu"

int main()
{
    cudaSetDevice(0);

    Reflection<float> test1(128);

    float* ptr = Host(test1);

    test1.Receive();

    ptr = Host(test1);

    return 0;
}