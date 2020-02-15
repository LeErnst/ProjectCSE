# --- general                                                                    
import logging                                                                   
import math                                                                      
import numpy
import random                                                                    
                                                                                 
# --- optical system and raytracing                                              
from pyrateoptics.raytracer.optical_system              import OpticalSystem     
from pyrateoptics.raytracer.optical_element             import OpticalElement    
from pyrateoptics.raytracer.localcoordinates            import LocalCoordinates  
from pyrateoptics.raytracer.surface                     import Surface           
from pyrateoptics.raytracer.surface_shape               import Conic             
from pyrateoptics.raytracer.aperture                    import CircularAperture  
from pyrateoptics.raytracer.material.material_isotropic import ConstantIndexGlass
from pyrateoptics.sampling2d                            import raster            
from pyrateoptics.raytracer.ray                         import RayPath           
from pyrateoptics.raytracer.ray                         import RayBundle         
from pyrateoptics.raytracer.globalconstants             import canonical_ey      
from pyrateoptics.raytracer.globalconstants             import degree            
                                                                                 
# --- optical system analysis                                                    
from pyrateoptics.raytracer.analysis.optical_system_analysis import \
                                                            OpticalSystemAnalysis
from pyrateoptics.raytracer.analysis.surface_shape_analysis  import ShapeAnalysis
# --- optimization                                                               
from pyrateoptics.optimize.optimize          import Optimizer                    
from pyrateoptics.optimize.optimize_backends import (ScipyBackend,               
                                                     Newton1DBackend,            
                                                     ParticleSwarmBackend,       
                                                     SimulatedAnnealingBackend)  
                                                                                 
# --- auxiliarys                                                                 
from auxiliary_functions import error2squared, error1, eval_h, eval_c 



def get_stochastic_grad(optimi, initialbundle, sample_param='wave'):

    wavel = len(initialbundle[0])
    numrays = initialbundle[0][0].x.shape[2]
    # determine the range from which numbers can be drawn
    sample_range = len(initialbundle)

    if (sample_param == 'wave'):
        sample_range *= wavel

    if (sample_param == 'ray'):
        sample_range *= wavel*numrays

    def stochastic_grad(func, x, h, *args):
        dim = len(x)
        sgrad = numpy.zeros_like(x)
        E = numpy.eye(dim,dim)
        # draw a number from the range [0, sample_range-1] (it is an array-index)
        sample_num = random.randint(0, sample_range-1)

        # set the meritparams in Optimizer-class, such that the meritfunctionrms
        # can figure out which initialbundle has been drawn
        optimi.meritparameters['sample_num'] = sample_num

        # calculate the stochastic gradient
        for i in range(dim):
            sgrad[i] = (func(x+h*E[i,:]) - func(x-h*E[i,:]))/(2*h)

        if (len(args) > 0):
            sgrad2 = numpy.zeros_like(x)
            for i in range(dim):
                sgrad2[i] = (func(args[0]+h*E[i,:])-func(args[0]-h*E[i,:]))/(2*h)
            sgrad2 *= sample_range

        # asure that the expectation of the stochastic gradient is the gradient
        sgrad *= sample_range

        # reset the meritparams in Optimizer-class, such that the full merit-
        # functionrms is evaluated
        optimi.meritparameters = {}

        if (len(args) > 0):
            return [sgrad,sgrad2]
        else:
            return sgrad
    
    return stochastic_grad


def grad(func, x, h):
    dim = len(x)
    grad = numpy.empty_like(x)
    E = numpy.eye(dim,dim)

    # calculate the gradient with finit differences
    for i in range(dim):
        grad[i] = (func(x+h*E[i,:]) - func(x-h*E[i,:]))/(2*h)

    return grad

def grad_pen(x, bdry, tau) :
    # gradient of the penalty term
    dim = len(x)
    grad = numpy.zeros(dim)
    h_x = eval_h(x, bdry)

    for i in range(2*dim) :
        if h_x[i] != 0 :
            grad[i/2] = tau*h_x[i]*numpy.power(-1, i+1)

    return grad


def grad_lag(x, bdry, tau, lam) :
    # gradient of the lambda-term + gradient of the penalty-term
    dim = len(x)
    grad = numpy.zeros(dim)
    h_x = eval_h(x, bdry)

    for i in range(2*dim) :
        if h_x[i] != 0 :
            grad[i/2] = (lam[i] + tau*h_x[i])*numpy.power(-1, i+1)

    return grad


def grad_log(x, bdry, my) :
    dim = len(x)
    grad = numpy.zeros(dim)
    c_x = eval_c(x, bdry)

    for i in range(2*dim) :
        if c_x[i] != 0 :
            grad[i/2] += (my/c_x[i])*numpy.power(-1,i)

            #now it is -1^i, as c(x) is now c(x)>=0, and not c(x)=0 as it was
            #before in grad_lag and grad_pen!

    return grad

def hessian(func,x,h=numpy.sqrt(numpy.finfo(float).eps)):
    dim = len(x)
    hessian = numpy.empty([dim,dim])
    E = numpy.eye(dim,dim)

    for i in range(dim):
        for j in range(dim):
            hessian[i,j] = (func(x+h*E[i,:]+h*E[j,:])-func(x+h*E[i,:]) - \
                            func(x+h*E[j,:])+func(x)) / (h**2)
    return hessian

