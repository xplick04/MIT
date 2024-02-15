/**
 * @file BatchMandelCalculator.cc
 * @author Maxim Plička <xplick04@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 01.11.2023
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <mm_malloc.h>
#include <string.h>


#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

#define BATCH_SIZE 128

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	dataBuff = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	imagBuff = (float *)_mm_malloc(BATCH_SIZE * sizeof(float), 64);
	processedBuff = (int *)_mm_malloc(BATCH_SIZE * sizeof(int), 64);
	realBuff = (float *)_mm_malloc(BATCH_SIZE * sizeof(float), 64);
	realBuffCopy = (float *)_mm_malloc(BATCH_SIZE * sizeof(float), 64);

	// setting default value
	for (int i = 0; i < height * width; i++)
	{
		*(dataBuff + i) = limit;
	}
}

BatchMandelCalculator::~BatchMandelCalculator() {
	_mm_free(dataBuff);
	_mm_free(realBuff);
	_mm_free(realBuffCopy);
	_mm_free(imagBuff);
	_mm_free(processedBuff);

	dataBuff = NULL;
	realBuff= NULL;
	realBuffCopy= NULL;
	imagBuff = NULL;
	processedBuff= NULL;

}


int * BatchMandelCalculator::calculateMandelbrot () {
 	float y;
	float x;
	float r2;
	float i2;
	int processedDone = 1;
	float mod;
	float div;
	int h = width * int(height / 2);

	float x_start = float(this->x_start);
	float y_start = float(this->y_start);
	float dx = float(this->dx);
	float dy = float(this->dy);

	// redeclaration, aligned() needs to see declarations to compile
	float *realBuff = this->realBuff;
	float *realBuffCopy = this->realBuffCopy;
	float *imagBuff = this->imagBuff;
	int *processedBuff = this->processedBuff;
	int *dataBuff = this->dataBuff;

	for (int b = 0; b < h; b += BATCH_SIZE)
	{
		mod = b % width;
		div = b / width;
		x = x_start + (mod) * dx; // batch start real value
		y = y_start + (div) * dy; // batch start imaginary value

		// preparing batch data
		#pragma omp simd aligned(processedBuff, realBuff, realBuffCopy, imagBuff : 64) 
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			processedBuff[i] = 0;
			realBuff[i] = x;
			realBuffCopy[i] = x;	// copy for original "real" value
			imagBuff[i] = y;
			x+=dx;
		}

		// computing batch data and writing into final data buffer
		for (int k = 0; k < limit; ++k)
		{

			#pragma omp simd simdlen(64) aligned(processedBuff, realBuff, realBuffCopy, imagBuff, dataBuff : 64)
			for (int i = 0; i < BATCH_SIZE; i++)
			{
				if (processedBuff[i]) continue;
				r2 = realBuff[i] * realBuff[i];
				i2 = imagBuff[i] * imagBuff[i];
				
				if (r2 + i2 > 4.0f)
				{
					processedBuff[i] = 1;
					*(dataBuff + b + i) = k;	// upper half
					*((dataBuff + (width) * (height - 1 - int(div))) + int(mod) + i) = k;	// lower half
				}

				imagBuff[i] = 2.0f * realBuff[i] * imagBuff[i] + y;
				realBuff[i] = r2 - i2 + realBuffCopy[i];
			}

			processedDone = 1;
			
			// break "mandelbrot" loop if whole batch was already computed
			#pragma omp simd simdlen(64) reduction(&: processedDone) aligned(processedBuff : 64)
			for (int i = 0; i < BATCH_SIZE; i++)
			{
				processedDone &= processedBuff[i];
			}
			if (processedDone) break;
		}
	}

	return dataBuff;
}
