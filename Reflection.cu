#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

////////////////////////////////////////////////////////////////////////

template <typename Type>

class Reflection
{
private:
	
	unsigned int Size = 0;

	void Clear()
	{
		host = nullptr;
		device = nullptr;
		
		Size = 0;
	}

public:

	Type* host = nullptr;
	Type* device = nullptr;

	Reflection::Reflection()
	{
		Clear();
	}

	Reflection::Reflection(const unsigned int count)
	{
		const unsigned int size = count * sizeof(Type);

		if(cudaMalloc(&device, size) != cudaSuccess)
		{
			Clear();
			return;
		}

		if(cudaMemset(device, 0, size) != cudaSuccess)
		{
			cudaFree(device);
			Clear();
			return;
		}

		host = new Type[count];

		memset(host, 0, size);

		Size = size;
	}

	Reflection::Reflection(Type* buffer, const unsigned int count)
	{
		const unsigned int size = count * sizeof(Type);

		if(cudaMalloc(&device, size) != cudaSuccess)
		{
			Clear();
			return;
		}

		if(cudaMemset(device, 0, size) != cudaSuccess)
		{
			cudaFree(device);
			Clear();
			return;
		}

		host = new Type[count];

		memcpy(host, buffer, size);

		Size = size;
	}

	Reflection::~Reflection()
	{
		Free();
	}

	bool IsValid()
	{
		if(Size == 0)
		{
			return false;
		}

		if(device == nullptr || host == nullptr)
		{
			return false;
		}

		return true;
	}

	void Free()
	{
		if(Size)
		{
			if(host != nullptr)
			{
				delete []host;
			}

			if(device != nullptr)
			{
				cudaFree(device);
			}
		}

		Clear();
	}

	unsigned int GetSize()
	{
		return Size;
	}

	bool Send()
	{
		if(IsValid())
		{
			return cudaMemcpy(device, host, Size, cudaMemcpyHostToDevice) == cudaSuccess;
		}

		return false;
	}

	bool Receive()
	{
		if(IsValid())
		{
			return cudaMemcpy(host, device, Size, cudaMemcpyDeviceToHost) == cudaSuccess;
		}

		return false;
	}
};

////////////////////////////////////////////////////////////////////////

template <typename Type>

Type* Device(Reflection<Type>& reflection)
{
	return reflection.device;
}

////////////////////////////////////////////////////////////////////////

template <typename Type>

Type* Host(Reflection<Type>& reflection)
{
	return reflection.host;
}

////////////////////////////////////////////////////////////////////////






