
# coding: utf-8

# In[1]:

import underworld as uw
import glucifer
from underworld import function as fn
import numpy as np


# In[5]:

def convection_vels(res):
    mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
    elementRes  = (res, res), 
    minCoord    = (0., 0.), 
    maxCoord    = (1., 1.))
    velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
    pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
    
    # Set viscosity to be a constant.
    viscosity = 1.
    # Rayleigh number.
    Ra = 1.0e2
    coord = fn.input()
    pertCoeff = fn.math.cos( 2.*np.pi * coord[0] ) * fn.math.sin( 1.*np.pi * coord[1] )
    
    # Construct our density function.
    densityFn = Ra * pertCoeff 
    buoyancyFn = densityFn*(0.,1.)
    
    iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
    jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
    
    # 2D velocity vector can have two Dirichlet conditions on each vertex, 
    # v_x is fixed on the iWalls (vertical), v_y is fixed on the jWalls (horizontal)
    velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                               indexSetsPerDof = (iWalls, jWalls) )
    
    stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = viscosity, 
                            fn_bodyforce  = buoyancyFn )

    # get the default stokes equation solver
    solver = uw.systems.Solver( stokes )
    solver.solve()
    
    return mesh, velocityField


# In[6]:

mesh, velocityField = convection_vels(16)


# In[7]:

fig = glucifer.Figure( figsize=(800,400) )
fig.append( glucifer.objects.Surface(mesh,fn.math.dot(velocityField, velocityField)) )
fig.append( glucifer.objects.VectorArrows(mesh, velocityField, scaling = 1e-1))
fig.append( glucifer.objects.Mesh(mesh) )
fig.save_database('test.gldb')


# In[ ]:



