/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Maxim Plička <xplick04@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    6.12.2023
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"
#include <omp.h>

//0      1      2      3     4     5     6 
//64 - > 32 - > 16 - > 8 - > 4 - > 2 - > 1
#define CUT_OFF 5

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{
    //mTriangles.reserve(mGridSize*mGridSize*mGridSize);
    for(int i = 0; i < omp_get_max_threads(); i++)
    {
        triangles.push_back(std::vector<Triangle_t>());
    }
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned totalTriangles = 0;

    #pragma omp parallel
    {
      #pragma omp single nowait
      { 
        totalTriangles = recursiveDecomposition(Vec3_t<float>(), mGridSize, field, 0);
      }
    }
    
    // Merge all triangles from all threads into one vector
    for (const auto& innerVector : triangles) 
    {
        std::copy(innerVector.begin(), innerVector.end(), std::back_inserter(mTriangles));
    }

    return totalTriangles;

}

unsigned TreeMeshBuilder::recursiveDecomposition(const Vec3_t<float> &offset, const float gridSize, const ParametricScalarField &field, int currRecNum)
{
    unsigned totalTriangles = 0;

    if(containsIsosurface(offset, gridSize, field))    // if the cube does not contain the isosurface
    {
        return 0;
    }

    if(gridSize <= 1)  // if the cube is the smallest possible
    {
        return buildCube(offset, field);
    }

    float newEdgeLen = gridSize / 2;

    for(int i = 0; i < 8; i++)   // for each child, 2^3 = 8
    {
        #pragma omp task shared(totalTriangles) firstprivate(i) final(currRecNum >= CUT_OFF) // create a task
        {
            Vec3_t<float> newOffset = offset;
            if(i & 1)   // if i is 1, 3, 5 or 7
            {
                newOffset.x += newEdgeLen;
            }
            if(i & 2)   // if i is 2, 3, 6 or 7
            {
                newOffset.y += newEdgeLen;
            }
            if(i & 4)   // if i is 4, 5, 6 or 7
            {
                newOffset.z += newEdgeLen;
            }

            #pragma omp atomic update
            totalTriangles += recursiveDecomposition(newOffset, gridSize / 2, field, currRecNum + 1);
        }
    }
    
    #pragma omp taskwait    // wait for all tasks to finish
    return totalTriangles;
}

bool TreeMeshBuilder::containsIsosurface(const Vec3_t<float> &offset,const float edgeLen, const ParametricScalarField &field)
{   
    float halfEdgeLen = edgeLen / 2.0f * mGridResolution; // half of the cube's edge length
    Vec3_t<float> cubeCenter(offset.x * mGridResolution + halfEdgeLen
                            ,offset.y * mGridResolution + halfEdgeLen
                            ,offset.z * mGridResolution + halfEdgeLen
                            );

    return evaluateFieldAt(cubeCenter, field) > mIsoLevel + sqrt(3.0f) * halfEdgeLen;   // f(p) > l + sqrt(3) * a / 2
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{

    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    triangles[omp_get_thread_num()].push_back(triangle);
}
