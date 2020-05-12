#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Reflection.cu"

int main()
{
    Reflection<float> test = Malloc<float>(128);



    return 0;
}