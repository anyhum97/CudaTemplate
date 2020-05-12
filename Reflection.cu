#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

template <typename Type>

struct Reflection
{
	Type* host = nullptr;
	Type* device = nullptr;

	unsigned int Size = 0;
	unsigned int IsValid = 0;
};

template <typename Type>

Reflection<Type> Malloc(const unsigned int count)
{
	Reflection<Type> reflection;

	const unsigned int size = count * sizeof(Type);

	if(cudaMalloc(&reflection.device, size) != cudaSuccess)
	{
		return reflection;
	}
	
	if(cudaMemset(reflection.device, 0, size) != cudaSuccess)
	{
		cudaFree(reflection.device);
		reflection.device = nullptr;
		return reflection;
	}

	reflection.host = new Type[count];

	memset(reflection.host, 0, size);

	reflection.size = size;

	return reflection;
}



