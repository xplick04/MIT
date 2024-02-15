/**
 * @file LineMandelCalculator.cc
 * @author Maxim Plička <xplick04@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 01.11.2023
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <mm_malloc.h>

#include <stdlib.h>

#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrijBaseSize, unsigned limit) :
	BaseMandelCalculator(matrijBaseSize, limit, "LineMandelCalculator")
{
	// 64 = CACHE line size
	dataBuff = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	processedBuff = (int *)_mm_malloc(width * sizeof(int), 64);
	imagBuff = (float *)_mm_malloc(width * sizeof(float), 64);
	realBuff = (float *)_mm_malloc(width * sizeof(float), 64);
	realBuffCopy = (float *)_mm_malloc(width * sizeof(float), 64);

	for (int i = 0; i < height * width; i++)
	{
		*(dataBuff + i) = limit;
	}
}

LineMandelCalculator::~LineMandelCalculator() {
	_mm_free(dataBuff);
	_mm_free(processedBuff);
	_mm_free(realBuff);
	_mm_free(realBuffCopy);
	_mm_free(imagBuff);

	dataBuff = NULL;
	processedBuff= NULL;
	realBuff= NULL;
	realBuffCopy = NULL;
	imagBuff = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
	float y = y_start;
	float x = x_start;
	float r2;
	float i2;
	int processedDone;
	int h = height/2;

	// redeclare doubles into floats
	float dx = float(this->dx);
	float dy = float(this->dy);
	float x_start = float(this->x_start);
	float y_start = float(this->y_start);

	// redeclaration, aligned() needs to see declarations to compile
	float *realBuff = this->realBuff;
	float *realBuffCopy = this->realBuffCopy;
	float *imagBuff = this->imagBuff;
	int *processedBuff = this->processedBuff;
	int *dataBuff = this->dataBuff;

    for (int i = 0; i < h; i++)
    {
		x = x_start;
		// preparing line data
		#pragma omp simd aligned(processedBuff, realBuff, realBuffCopy, imagBuff : 64)
        for (int j = 0; j < width; j++)
        {
			processedBuff[j] = 0;
            realBuff[j] = x;
			realBuffCopy[j] = x;	// copy for original "real" value
            imagBuff[j] = y;
			x += dx; 
        }

		// computing line data and writing into final data buffer
		for (int k = 0; k < limit; ++k)
        {

            #pragma omp simd simdlen(64) aligned(processedBuff, realBuff, realBuffCopy, imagBuff, dataBuff : 64)
			for (int j = 0; j < width; j++)
			{
				if(processedBuff[j]) continue;
				r2 = realBuff[j] * realBuff[j];
				i2 = imagBuff[j] * imagBuff[j];

				if (r2 + i2 > 4.0f)
				{
					processedBuff[j] = 1;
					*(dataBuff + i*width + j) = k;	// upper half
					*(dataBuff + (height - 1 - i)*width + j) = k;	// lower half
				}

				imagBuff[j] = 2.0f * realBuff[j] * imagBuff[j] + y;
				realBuff[j] = r2 - i2 + realBuffCopy[j];	
				
			}

			processedDone = 1;

			// break "mandelbrot" loop if whole batch was already computed 
            #pragma omp simd simdlen(64) reduction(&:processedDone) aligned(processedBuff : 64)
            for (int j = 0; j < width; j++)
			{
				processedDone &= processedBuff[j];
			}
            if (processedDone) break;
        }
		y += dy;
    }
    return dataBuff;
}