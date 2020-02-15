# file to generate plots

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
                                       sgdRestart,
                                       adam,
                                       adamRestart,
                                       adamax,
                                       adagrad,
                                       adadelta,
                                       get_scipy_stochastic_hybrid,
                                       SdLBFGS,
                                       gd_splitted_stepsize,
                                       nag_splitted)
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
# parameter which are used in the project report
# opt_func, whichmeritfunc, sample_param, methods, stepsize, gamma, beta1, beta2, epsilon, roh

# sgd:
# iteration number
#   vanilla : sgd1, bundle, stepsize=1e-9
#   momentum: sgd1, bundle, stepsize=1e-9, gamma=0.9
#   nag     : sgd1, bundle, stepsize=1e-9, gamma=0.9
# time
#   nag     : sgd1, bundle, stepsize=1e-9, gamma=0.9
#   nag     : sgd1, wave, stepsize=1e-9, gamma=0.9
#   momentum: sgd1, bundle, stepsize=1e-9, gamma=0.9

# adam: sgd1, bundle, stepsize=1e-2/1e-3, beta1=0.9, beta2=0.99, epsilon=1e-8 (betak=beta1/sqrt(lam))
#       sgd1, bundle, stepsize=1e-2, beta1=0.1/0.9, beta2=0.99, epsilon=1e-8 (betak=beta1)

# adamax: sgd1, bundle, stepsize=1e-4, beta1=0.1/0.9, beta2=0.1/0.99/0.999 

# adagrad: sgd1, bundle, stepsize=1e-2/1e-3/1e-4, epsilon=1e-3

# adadelta: sgd1, bundle, rho=0.1/0.5/0.9, epsilon=1e-8

opt_func         = adam
whichmeritfunc   = 'sgd1'
sample_paramlist = 'bundle'
methods_list     = ['nag']
maxiter          = 250
stepsize_list    = [1e-4]
gamma_list       = [0.9]
beta1_list       = [0.9]
beta2_list       = [0.999]
epsilon_list     = [1e-8]
roh_list         = [0.1]
legend_list      = ['']
xlabel = 'Iteration number'
ylabel = 'Meritfunction value'
fignum = 1
optsysfig = 2
penalty = True
penaltyVerz = True
i = 0

for stepsize in stepsize_list:
    for gamma in gamma_list:
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for epsilon in epsilon_list:
                    for roh in roh_list:
                        legend       = legend_list[i]
                        methods      = methods_list[0]
                        sample_param = sample_paramlist
                        i += 1

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
                        
                        (initialbundle, meritfunctionrms) = get_bundle_merit(osa, s, sysseq, 
                                                                             rays_dict,
                                                                             fi1.numrays, fi1.wavelengths, 
                                                                             whichmeritfunc=whichmeritfunc, 
                                                                             error='error2',
                                                                             sample_param=sample_param,
                                                                             penalty=penalty,
                                                                             penaltyVerz=penaltyVerz,
                                                                             f=100.0)
                        
                        
                        # ----- plot the original system
                        # --- set the plot setting
                        '''
                        pn = np.array([1, 0, 0])
                        up = np.array([0, 1, 0])
                        
                        fig = plt.figure(optsysfig)
                        ax1 = fig.add_subplot(211)
                        ax2 = fig.add_subplot(212)
                        ax1.axis('equal')
                        ax2.axis('equal')
                        '''
                        # --- plot the bundles and draw the original system
                        # first it is necessary to copy the initialbundle, as it's gonna be changed
                        # inside of the seqtrace function (called inside of plotBundles)
                        #testbundle = deepcopy(initialbundle)
                        #plotBundles(s, initialbundle, sysseq, ax1, pn, up)
                        
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
                        
                        
                        #hybrid_method = get_scipy_stochastic_hybrid(stocha_opt_func=sgd,
                        #                                            scipy_opt_func ='Newton-CG') 
                    
                        plotsettings = {'fignum'    : fignum,
                                        'title'     : '',
                                        'fonttitle' : '14',
                                        'xlabel'    : xlabel,
                                        'ylabel'    : ylabel,
                                        'fontaxis'  : 12,
                                        'legend'    : legend,
                                        'fontlegend': 12,
                                        'loclegend' : 'best',
                                        'xlim'      : 'auto',
                                        'ylim'      : 'auto',
                                        'xlog'      : True,
                                        'ylog'      : True,
                                        'xticks'    : False,
                                        'yticks'    : False,
                                        'axformat'  : 'sci',
                                        'grid'      : True,
                                        'linewidth' : 3,
                                        'linestyle' : '-',
#                                        'color'     : color,
                                        'marker'    : 'o',
                                        'markersize': 5,
                                        'save'      : False,
                                        'name'      : 'sgd.png',
                                        'show'      : False}
                        
                        # options for stochastic optimization method
                        # TODO: change the gradtol such that the stocha method and langrange/penalty 
                        #       have the same termination condition
                        # TODO: generalize the methods to plot more than one curve
                        options_s1 = {'gtol'    : 1e+8,
                                      'gradtol' : 500,
                                      'maxiter' : maxiter, 
                                      'stepsize': stepsize, 
                                      'beta1'   : beta1, 
                                      'beta2'   : beta2,
                                      'roh'     : roh,
                                      'epsilon' : epsilon,
                                      'gamma'   : gamma,
                                      'methods' : methods,
                                      'pathf'   : True,
                                      'plot'    : True,
                                      'plotset' : plotsettings}
                        
                        # options for deterministic optimization method
                        options_d = {'maxiter': 250, 
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
                        opt_backend_1 = ProjectScipyBackend(optimize_func=opt_func,
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

plt.show()

#*******************************************************************************
'''


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
'''
