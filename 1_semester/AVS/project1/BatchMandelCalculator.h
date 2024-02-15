/**
 * @file BatchMandelCalculator.h
 * @author Maxim Plička <xplick04@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 01.11.2023
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
	int* dataBuff;
	int* processedBuff;
	float* realBuff;
    float* realBuffCopy;
	float* imagBuff;
};

#endif