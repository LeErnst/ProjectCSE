# lewin has created this file for testing some algorithms and to not change the 
# original file (01_..), which is the main file for patricks changes (outsourcing
# some functions and so on). i will use this file only for testing 


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
from project_optimize_backends import (ProjectScipyBackend,
                                       test_minimize_neldermead,
                                       gradient_descent,
                                       sgd,
                                       adam,
                                       adamax,
                                       adagrad,
                                       adadelta,
                                       get_scipy_stochastic_hybrid,
                                       plot2d_meritfunction,
                                       gd_splitted_stepsize)

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
                                get_bdry, \
                                plot2d

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
rays_dict = fi1.get_rays_dict()

#rays_dict = {"startz":[0], "starty": [0], "radius": [16],
#             "anglex": [0., 0.1832595], 
#             "rasterobj":raster.RectGrid()}

wavelength = [0.5875618e-3, 0.4861327e-3, 0.6562725e-3]
numrays = 20
sample_param = 'bundle'

(initialbundle, meritfunctionrms) = get_bundle_merit(osa, s, sysseq, rays_dict,
                                                     numrays, wavelength, 
                                                     whichmeritfunc='sgd1', 
                                                     error='error2',
                                                     sample_param=sample_param,
                                                     penalty=True)


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


# --- define update function
def osupdate(my_s):
#   Update all coordinate systems during run
    my_s.rootcoordinatesystem.update()


#*******************************************************************************
# choose our own backend for testing some algos

# possible methodparams = {standard, penalty, penalty-lagrange, log}

# ---- gradient descent
# for problems increase the stepsize
#opt_backend = ProjectScipyBackend(optimize_func=gradient_descent,
#                                  methodparam='penalty-lagrange',
#                                  options={})

# ---- own neldermead
#opt_backend = ProjectScipyBackend(optimize_func=test_minimize_neldermead,
#                                  methodparam='penalty-lagrange',
#                                  options={'maxiter': 100 , 'xatol': 1e-5,\
#                                           'fatol': 1e-5})

#hybrid_method = get_scipy_stochastic_hybrid(stocha_opt_func=sgd,
#                                            scipy_opt_func ='Newton-CG') 

plotsettings = {'fig'      : 1,
                'title'    : r'p_{i}-meritfunction-plot',
                'xlabel'   : r'p_{i}',
                'ylabel'   : 'meritfunctionvalue',
                'legend'   : '',
                'ylog'     : False,
                'linestyle': '-o', 
                'save'     : False,
                'name'     : '',
                'show'     : False}

# options for stochastic optimization method
# TODO: change the gradtol such that the stocha method and langrange/penalty 
#       have the same termination condition
# TODO: generalize the methods to plot more than one curve
options_s1 = {'gtol'     : 1e+8,
              'maxiter'  : 150, 
              'stepsize' : 1e-7, 
              'beta1'    : 0.1, 
              'beta2'    : 0.99,
              'gradtol'  : 500,
              'roh'      : 0.1,
              'c'        : 1e-2,
              'delta_min': 1e-1,
              'delta_max': 1e+6,
              'epsilon'  : 1e-8,
              'gamma'    : 0.1,
              'methods'  : 'vanilla',
              'pathf'    : True,
              'plot'     : False,
              'plotset'  : plotsettings}

# options for deterministic optimization method
options_d = {'maxiter': 150, 
             'xtol'   : 1e-9, 
             'ftol'   : 1e-5,
             'gtol'   : 1e+1,
             'disp'   : True}

# options for hybrid optimization method
# attention: for a hybrid optimization method you need to remove the gtol option
#            in options_s
#options={'gtol'     : 1e+3,
#         'plot'     : True, 
#         'options_d': options_d,
#         'options_s': options_s}

# ---- stochastic gradient descent
opt_backend_1 = ProjectScipyBackend(optimize_func=gd_splitted_stepsize,
                                    methodparam='penalty-lagrange',
                                    stochagradparam=True,
                                    options=options_s1)

# ----- create optimizer object
optimi_1 = Optimizer(s,
                     meritfunctionrms,
                     backend=opt_backend_1,
                     updatefunction=osupdate)

fi1.store_data(s)
fi1.write_to_file()

# ----- start optimizing
opt_backend_1.update_PSB(optimi_1)

stochagrad_1 = get_stochastic_grad(optimi_1, initialbundle, sample_param=sample_param)
opt_backend_1.stochagrad = stochagrad_1

s1 = optimi_1.run()

#*******************************************************************************
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
fi1.store_data(s)
fi1.write_to_file()

