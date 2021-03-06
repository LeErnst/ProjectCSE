# template to create a optical system which is optimized
# including the necessary classes, functions and libraries
# by patrick, leandro, michael, lewin


# lewin has created this file for testing some algorithms and to not change the 
# original file (01_..), which is the main file for patricks changes (outsourcing
# some functions and so on). i will use this file only for testing until the 
# main.py script is made and i will not change any functions or classes which
# have patrick/leandro created


# --- general
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

# --- copy function for copying the initialbundle
from copy import deepcopy

# --- optical system and raytracing
from pyrateoptics.raytracer.optical_system              import OpticalSystem
from pyrateoptics.raytracer.optical_element             import OpticalElement
from pyrateoptics.raytracer.localcoordinates            import LocalCoordinates
from pyrateoptics.raytracer.surface                     import Surface
from pyrateoptics.raytracer.surface_shape               import Conic
from pyrateoptics.raytracer.aperture                    import CircularAperture
from pyrateoptics.raytracer.material.material_isotropic import\
                                                              ConstantIndexGlass
from pyrateoptics.sampling2d                            import raster
from pyrateoptics.raytracer.ray                         import RayPath
from pyrateoptics.raytracer.ray                         import RayBundle
from pyrateoptics.raytracer.globalconstants             import canonical_ey 
from pyrateoptics.raytracer.globalconstants             import degree 

# --- optical system analysis
from pyrateoptics.raytracer.analysis.optical_system_analysis import\
                                                        OpticalSystemAnalysis
from pyrateoptics.raytracer.analysis.surface_shape_analysis  import\
                                                        ShapeAnalysis
# --- optimization
from pyrateoptics.optimize.optimize          import Optimizer
from pyrateoptics.optimize.optimize_backends import (ScipyBackend,
                                                     Newton1DBackend,
                                                     ParticleSwarmBackend,
                                                     SimulatedAnnealingBackend)
from project_optimize_backends import (ProjectScipyBackend,\
                                       test_minimize_neldermead,\
                                       sgd,\
                                       gradient_descent)
# --- debugging 
from pyrateoptics import listOptimizableVariables

# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig()

# --- auxiliary functions
from auxiliary_functions import calculateRayPaths,\
                                error2squared,\
                                error1,\
                                setOptimizableVariables,\
                                calcBundleProps,\
                                plotBundles,\
                                inout,\
                                plotSpotDia, \
                                get_bdry
# --- meritfunction and initialbundle
from aux_merit_bundle import buildInitialbundle, get_bundle_merit

# --- derivatives
from derivatives import get_stochastic_grad




#####################################NeuAnfang##################################
#create inout object for all io stuff
fi1=inout()

#create optical system
s = OpticalSystem.p()

#create for each surface a coordinate system which includes loading the surfaces
cs=fi1.create_coordinate_systems(s)

#create optical element
elem1 = OpticalElement.p(cs[0], name="elem1")

#create surface objects
surf=fi1.create_surfaces(cs)

#create material
fi1.create_material(cs, elem1, surf)
#######################################Neuende##################################

# ----------- assemble optical system
s.addElement(elem1.name, elem1)


# II---------------------- optical system analysis
# --- 1. elem

sysseq=fi1.get_sysseq(elem1);

# ----------- define optical system analysis object
osa = OpticalSystemAnalysis(s, sysseq)




# III ----------- defining raybundles for optimization and plotting 
rays_dict=fi1.get_rays_dict()

(initialbundle, meritfunctionrms) = get_bundle_merit(osa, s, sysseq, rays_dict,
                                    fi1.numrays, fi1.wavelengths, 
                                    whichmeritfunc='standard', 
                                    error='error2')


# ----- plot the original system
# --- set the plot setting
pn = np.array([1, 0, 0])
up = np.array([0, 1, 0])

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.axis('equal')
ax2.axis('equal')

# --- plot the bundles and draw the original system
# first it is necessary to copy the initialbundle, as it's gonna be changed
# inside of the seqtrace function (called inside of plotBundles)
testbundle = deepcopy(initialbundle)
plotBundles(s, initialbundle, sysseq, ax1, pn, up)


# IV ----------- optimization
# ----- define optimizable variables
#######################################NeuAnfang################################
fi1.setup_variables(s,elem1.name)
#######################################NeuEnde##################################

# --- print table of optimizable variables
# this is actually quite nice to get a look at the vars, especially at the end
print("listedervariablen")
listOptimizableVariables(s, filter_status='variable', max_line_width=1000)

# --- define update function
def osupdate(my_s):
#   Update all coordinate systems during run
    my_s.rootcoordinatesystem.update()


# ----- choose the backend


#*******************************************************************************
#****ALS FUNKTION AUSLAGERN???**************************************************
#*******************************************************************************
# choose our own backend for testing some algos

# possible methodparams = {standard, penalty, penalty-lagrange, log}

# for problems increase the stepsize
opt_backend = ProjectScipyBackend(optimize_func='cg',#differential-evolution',#nelder-mead',#test_minimize_neldermead,
                                  methodparam='penalty-lagrange',
                                  options={'maxiter': 150, 'gtol':1e-2})#,
                                            #'xatol': 1e-14,
                                            #'fatol': 1e-14})

# ----- create optimizer object
optimi = Optimizer(s,
                   meritfunctionrms,
                   backend=opt_backend,
                   updatefunction=osupdate)

#bdry = get_bdry(optimi)

#Warum dreimal?!?
fi1.store_data(s)   #bei jedem aufruf werden die variablen gespeichert
fi1.store_data(s)
fi1.store_data(s)
fi1.write_to_file() #am ende einmal in ne datei schreiben

## ----- start optimizing
opt_backend.update_PSB(optimi)
s = optimi.run()
print('ANZAHL FUNKTIONSAUFRUFE GESAMT:')
print(optimi.NoC)
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************



# print the final simplex, which are the meritfunction values
# print("f simplex: ", opt_backend.res.final_simplex[1])
# print("iterNum = ", opt_backend.res.nit)
#
#
#
## V----------- plot the optimized system
#
## --- plot the bundles and draw the system
plotBundles(s, testbundle, sysseq, ax2, pn, up)
##
# --- draw spot diagrams
# plotSpotDia(osa, numrays, rays_dict, wavelength)


# get a look at the vars
ls=listOptimizableVariables(s, filter_status='variable', max_line_width=1000)

plt.show()
