/**
 * @file    tree_mesh_builder.h
 *
 * @author  Maxim Plička <xplick04@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    6.12.2023
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"
#include <vector>

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const 
    {
        return mTriangles.data(); 
    }
    unsigned recursiveDecomposition(const Vec3_t<float> &offset, const float gridSize, const ParametricScalarField &field, int currRecNum);
    bool containsIsosurface(const Vec3_t<float> &offset,const float gridSize, const ParametricScalarField &field);
    
    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles
    std::vector<std::vector<Triangle_t>> triangles; ///< Temporary array of triangles
};

#endif // TREE_MESH_BUILDER_H
