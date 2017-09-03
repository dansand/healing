
# coding: utf-8

# Thermal Convection
# ======
# 
# This example solves 2D dimensionless isoviscous thermal convection with a Rayleigh number of $10^4$, see Blankenbach *et al.* 1989 for details.
# 
# **This example introduces:**
# 1. Setting up material parameters and functions.
# 2. Setting up and solving systems, Stokes and Advection Diffusion.
# 3. The time stepping loop.
# 4. Plotting with glucifer.
# 
# **Keywords:** material parameters, Stokes system, advective diffusive systems
# 
# **References**
# 
# B. Blankenbach, F. Busse, U. Christensen, L. Cserepes, D. Gunkel, U. Hansen, H. Harder, G. Jarvis, M. Koch, G. Marquart, D. Moore, P. Olson, H. Schmeling and T. Schnaubelt. A benchmark comparison for mantle convection codes. Geophysical Journal International, 98, 1, 23–38, 1989
# http://onlinelibrary.wiley.com/doi/10.1111/j.1365-246X.1989.tb05511.x/abstract

# In[1]:

import underworld as uw
from underworld import function as fn
import glucifer
import math


# Setup parameters
# -----

# In[2]:

# Set simulation box size.
boxHeight = 1.0
boxLength = 2.0
# Set the resolution.
res = 64
# Set min/max temperatures.
tempMin = 0.0
tempMax = 1.0


# Create mesh and variables
# ------
# 
# The mesh object has both a primary and sub mesh. "Q1/dQ0" produces a primary mesh with element type Q1 and a sub-mesh with elements type dQ0. Q1 elements have nodes at the element corners, dQ0 elements have a single node at the elements centre.

# In[3]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (2*res, res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight))


# Create mesh variables.  Note the pressure field uses the sub-mesh. 

# In[4]:

velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

# Initialise values
velocityField.data[:]       = [0.,0.]
pressureField.data[:]       = 0.
temperatureDotField.data[:] = 0.


# In[5]:

# now add a variable
meshvar = uw.mesh.MeshVariable(mesh,1)
# set the mesh variable to the processor rank!
# note that the numpy view is only to local data!
meshvar.data[:] = uw.rank()


# Set up material parameters and functions
# -----
# 
# Set functions for viscosity, density and buoyancy force. These functions and variables only need to be defined at the beginning of the simulation, not each timestep.

# In[6]:

# Set viscosity to be a constant.
viscosity = 1.

# Rayleigh number.
Ra = 1.0e4

# Construct our density function.
densityFn = Ra * temperatureField

# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
z_hat = ( 0.0, 1.0 )

# Now create a buoyancy force vector using the density and the vertical unit vector. 
buoyancyFn = densityFn * z_hat


# Create initial & boundary conditions
# ----------
# 
# Set a sinusoidal perturbation in the temperature field to seed the onset of convection.

# In[7]:

pertStrength = 0.2
deltaTemp = tempMax - tempMin
for index, coord in enumerate(mesh.data):
    pertCoeff = math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
    temperatureField.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
    temperatureField.data[index] = max(tempMin, min(tempMax, temperatureField.data[index]))


# Set top and bottom wall temperature boundary values.

# In[8]:

for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin


# Construct sets for ``I`` (vertical) and ``J`` (horizontal) walls.

# In[9]:

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]


# Create Direchlet, or fixed value, boundary conditions. More information on setting boundary conditions can be found in the **Systems** section of the user guide.

# In[10]:

# 2D velocity vector can have two Dirichlet conditions on each vertex, 
# v_x is fixed on the iWalls (vertical), v_y is fixed on the jWalls (horizontal)
velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls, jWalls) )

# Temperature is held constant on the jWalls
tempBC = uw.conditions.DirichletCondition( variable        = temperatureField, 
                                           indexSetsPerDof = (jWalls,) )


# **Render initial conditions for temperature**
# 

# In[11]:

#figtemp = glucifer.Figure( figsize=(800,400) )
#figtemp.append( glucifer.objects.Surface(mesh, temperatureField, colours="blue white red") )
#figtemp.append( glucifer.objects.Mesh(mesh) )
#figtemp.show()


# ## Tools for smoothing and density of swarms
# 
# This machinery is supposed to work in parallel, and make use of the shadow particles
# However, we may need to be careful when multiple procs at thinning / thickening swarms, and the shadow particle info could be changing. 
# 
# 
# 

# In[12]:

from unsupported_dan.interfaces.marker2D import markerLine2D

from unsupported_dan.interfaces.smoothing2D import *


import numpy as np


# In[13]:

##Set up the starting coords for the passive swarm


targetDist = 2./400
startDist = 2./400

dl = targetDist/2.
du = (2.*(targetDist-dl))

lowDist = targetDist - dl
upDist = targetDist + du

np.random.seed(22)

xs = np.arange(0.1,mesh.maxCoord[0] - mesh.minCoord[0] - 0.1, startDist)

xs += 0.01*np.random.rand(xs.shape[0])

from matplotlib.pylab import shuffle
shuffle(xs)

##Let's mix up the coordinates



ys = 0.3*np.cos(np.pi*xs) + 0.5 + 0.2*np.cos(4*np.pi*xs) + 0.05*np.cos(12*np.pi*xs)  
ys += 0.01*np.random.rand(xs.shape[0])


Line = markerLine2D(mesh, velocityField, xs, ys, 0., 1.)

Line2 = markerLine2D(mesh, velocityField, xs, ys, 0., 1.)


# In[14]:

lowDist,upDist,targetDist


# In[15]:

#%pylab inline
#plt.plot(Line.swarm.particleCoordinates.data[:,0])


# In[16]:

# plot figure
figtemp = glucifer.Figure( figsize=(800,400) )
figtemp.append( glucifer.objects.Surface(mesh, temperatureField, colours="blue grey red", colourBar=False) )
figtemp.append( glucifer.objects.Points(Line.swarm,pointSize=4,colourBar=False ))
figtemp.append( glucifer.objects.Points(Line2.swarm,pointSize=7,colourBar=False,colour='white' ))


#figtemp.show()


# In[ ]:




# ## Test

# In[17]:

Line.rebuild()

if Line.swarm.particleCoordinates.data.shape[0]:

    

    A1 = neighbour1Matrix(Line)
    A2 = neighbour2Matrix(Line)
    A2 = neighbour2Matrix2(Line, angle = 45., k= 10)

    A = A1 + A2

    #Test the basic matrices we'll need
    L = laplacianMatrix(Line, A)
    P = pairDistanceMatrix(Line)
    R = neighboursAngleMatrix(Line)
    



# In[19]:

#This is the matrix that contains each particles 2 nearest neighbours.
#For some particles - ike end particles - only one neighbour will appear
#All we do here is add the nearest neighbout and the second-nearest
#the second-nearest should have distance-angle weighting applied

def buildA(markerLine, k):
    A1 = neighbour1Matrix(markerLine)
    #A2 = neighbour2Matrix(markerLine, k = k)
    A2 = neighbour2Matrix2(markerLine, angle = 40., k= k)


    return A1 + A2
    


# In[20]:

def shadowMask(markerLine):
    """
    This needs doing better
    Al I want to do is test membership of local processor swarm, 
    with the local + shadow space swarm
    
    """
    
    
    all_particle_coords = markerLine.kdtree.data
    m1 = np.in1d(markerLine.swarm.particleCoordinates.data[:,0], all_particle_coords[:,0])
    m2 = np.in1d(markerLine.swarm.particleCoordinates.data[:,1], all_particle_coords[:,1])

    mask = []
    for i in range(len(m1)):
        if m1[i] == True == m2[i]:
            mask.append(True)
        else:
            mask.append(False)
    
    
    return np.array(mask)


# In[20]:

#mesh.maxCoord


# ## Particle addition

# In[21]:

def particlesToAdd(markerLine, A, _dist, _updist = False):
    
    all_particle_coords = markerLine.kdtree.data
    
    
    #We want only the lower half of the matrix
    #including the upper half would add particles twice
    Alow = np.tril(A)
    
    pd = pairDistanceMatrix(markerLine)
    
    #Here is the distance mask
    if _updist:
        pdMask = np.logical_and(pd > _dist, pd < _updist)
    else:
        pdMask = pd > _dist
    
    #We only want to choose those particles that have two nearest neighbours (this hopefully excludes endpoints)
    mask = np.where(A.sum(axis=1) == 1)
    #Set those rows to zero
    pdMask[mask,:]  = 0 
    
    
    AF = Alow*pdMask
    
    uniques = np.transpose(np.nonzero(AF))
    #First, store a complete copy of the new particle positions (mean pair positions)
    newPoints = np.copy(0.5*(all_particle_coords[uniques[:,0]] + all_particle_coords[uniques[:,1]]))
    
    return newPoints
    


# In[22]:

#experimenting to find a safe way of applying these in parallel

if Line.swarm.particleCoordinates.data.shape[0]:
    A = buildA(Line, k =7)
    newPoints = particlesToAdd(Line, A, _dist=upDist, _updist=8.*upDist)
else:
    newPoints = np.array([[],[]]).T
    
print("added {} particles to Swarm".format(newPoints.shape[0]))
Line.add_points(newPoints[:,0], newPoints[:,1])
Line.rebuild()


# ## Particle deleting

# In[17]:

def particlesToCull(markerLine, A, _dist, fac=0.1):
    
    if markerLine.swarm.particleCoordinates.data.shape[0] > 2:
    
    
        pd = pairDistanceMatrix(markerLine)

            #mask for particles on local proc
        lpmask = shadowMask(markerLine)

        pdMask = (pd < _dist)[np.ix_(lpmask, lpmask)]




        #pdMask will generally have the diagonals in it, we want to remove these
        #diagIds = np.array(zip(np.arange(markerLine.kdtree.data.shape[0]), np.arange(markerLine.kdtree.data.shape[0])))
        #pdMask[diagIds[:,0], diagIds[:,1]] = False
        np.fill_diagonal(pdMask, False)


        #We're only interested in the part of matrix correspoding to neighbours
        pdMask = np.logical_and(pdMask, A[np.ix_(lpmask, lpmask)].astype('bool'))

        #We only want to choose those particles that have two nearest neighbours (this hopefully excludes endpoints)
        mask = np.where(A[np.ix_(lpmask, lpmask)].sum(axis=1) == 1)
        #Set those rows to zero
        pdMask[mask,:]  = 0 


        uniques = np.transpose(np.nonzero(pdMask))

        uniques = np.array(list(set(uniques[:,0])))

        #simple way to randomly choose points to cull
        np.random.shuffle(uniques)
        index = int(uniques.shape[0]*fac)
        uniques = uniques[:index]
        
        
        

        if uniques.shape[0] > 0:

            return uniques
        
        else:
            return  []
        
    else:
        return  []


# In[26]:

if Line.swarm.particleCoordinates.data.shape[0]:
    A = buildA(Line,k =7)
    toGo = particlesToCull(Line, A,lowDist, fac=0.2)
    
else:
    toGo = []
    
with Line.swarm.deform_swarm():
    Line.swarm.particleCoordinates.data[toGo] = (99999, 99999)

print("removed {} particles from Swarm".format(len(toGo)))

Line.rebuild()


# In[25]:

#Line.swarm.particleCoordinates.data[[]]


# ## Particle Smoothing

# In[27]:

def buildA_L(markerLine, k):
    A1 = neighbour1Matrix(markerLine)
    A2 = neighbour2Matrix(markerLine, k = k)
    #A2 = neighbour2Matrix2(Line, angle = 40., k= k)
    
    return A1 + A2


def laplaceSmooth(markerLine,A, k,  lam):
    """
    this includes my current solution for utilising the shadow space data
    We build th Laplcian using the full particle coordinate data (accessed through kdtree.data):
    our update looks like:
    lam * np.dot(L,Line.kdtree.data[mask]
    mask returns ony those particles on the local processor, hence on LHS:
    swarm.particleCoordinates.data[:]
    """   
    
   
    #rebuild A
    #A = buildA_L(markerLine,k)
    #Build Laplacian operator
    L = laplacianMatrix(markerLine, A)
    #build mask
    mask = shadowMask(markerLine)
    
    dl = lam * np.dot(L,markerLine.kdtree.data)
        
    #print(dl[mask].shape, markerLine.swarm.particleCoordinates.data.shape)
    #print(mask.shape)
    
    #with markerLine.swarm.deform_swarm():
    #    markerLine.swarm.particleCoordinates.data[:] -= dl[mask,:]
    
    return dl[mask]
    
    #
    #markerLine.swarm.update_particle_owners()


# lam = 0.0025*(mesh.maxCoord[0] - mesh.minCoord[0])
# 
# for i in range(100):
#     if Line.swarm.particleCoordinates.data.shape[0]:
#         A = buildA(Line,k =7)
#         Dl = laplaceSmooth(Line,A, 5,  lam)
#     else:
#         Dl = 0.0 
# 
#     with Line.swarm.deform_swarm():
#         Line.swarm.particleCoordinates.data[:] -= Dl
#     
#     Line.rebuild()

# ## Package it up

# In[31]:

def smoother():
    
    #Add particles

    if Line.swarm.particleCoordinates.data.shape[0]:
        A = buildA(Line, k =7)
        newPoints = particlesToAdd(Line, A, _dist=upDist, _updist=8.*upDist)
    else:
        newPoints = np.array([[],[]]).T

    print("added {} particles to Swarm".format(newPoints.shape[0]))
    Line.add_points(newPoints[:,0], newPoints[:,1])
    Line.rebuild()
    
    #Laplace smooth
    lam = 0.002*(mesh.maxCoord[0] - mesh.minCoord[0])

    for i in range(50):
        if Line.swarm.particleCoordinates.data.shape[0]:
            A = buildA(Line,k =7)
            Dl = laplaceSmooth(Line,A, 5,  lam)
        else:
            Dl = 0.0 

        with Line.swarm.deform_swarm():
            Line.swarm.particleCoordinates.data[:] -= Dl

        Line.rebuild()
        
    #cull
    
    for i in range(5):
        if Line.swarm.particleCoordinates.data.shape[0]:
            A = buildA(Line,k =7)
            toGo = particlesToCull(Line, A,lowDist, fac=0.2)

        else:
            toGo = []

        with Line.swarm.deform_swarm():
            Line.swarm.particleCoordinates.data[toGo] = (99999, 99999)

        print("removed {} particles from Swarm".format(len(toGo)))

        Line.rebuild()


# In[34]:

smoother()


# ## Viz

# In[35]:

fullpath ='./'

store1 = glucifer.Store(fullpath + 'subduction1.gldb')


## plot figure
fig1= glucifer.Figure(store1, figsize=(800,400) )
fig1.append(glucifer.objects.Surface(mesh,meshvar, valueRange=[0,uw.nProcs()]))
fig1.append( glucifer.objects.Points(Line.swarm,pointSize=4,colourBar=False ))
fig1.append( glucifer.objects.Points(Line2.swarm,pointSize=7,colourBar=False, colour='white' ))

#fig1.save_database('test.gldb')


# ## Stokes

# In[36]:

stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = viscosity, 
                            fn_bodyforce  = buoyancyFn )

# get the default stokes equation solver
solver = uw.systems.Solver( stokes )


advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField, 
                                         fn_diffusivity = 1.0, 
                                         conditions     = tempBC )


# ## Run

# In[37]:

# define an update function
def update():
    # Retrieve the maximum possible timestep for the advection-diffusion system.
    dt = advDiff.get_max_dt()
    # Advect using this timestep size.
    advDiff.integrate(dt)
    
    Line.advection(dt)
    Line2.advection(dt)
    
    return time+dt, step+1


# In[38]:

# init these guys
time = 0.
step = 0


# In[ ]:

steps_end = 200

# perform timestepping
while step < steps_end:
    # Solve for the velocity field given the current temperature field.
    solver.solve()
    time, step = update()
    
    if step % 5 == 0:
        
        print("Local shape:", Line.swarm.particleCoordinates.data.shape[0])
        print("S shape:", Line.swarm.particleCoordinates.data.shape[0])
        
        #Add particles
        smoother()
    
    if step % 2 == 0:
        
        fullpath = './'
        store1.step = step
        fig1.save( fullpath + "Temp" + str(step).zfill(4))
        
    print("step: " + str(step))


# In[ ]:



