/**
 * @file LineMandelCalculator.h
 * @author Maxim Plička <xplick04@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 01.11.2023
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int *dataBuff;
    int *processedBuff;
	float *imagBuff;
	float *realBuff;
    float *realBuffCopy;
};