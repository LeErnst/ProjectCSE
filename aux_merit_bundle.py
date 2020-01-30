# --- general
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import sys
import copy

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
from auxiliary_functions import error2squared, error1


def buildInitialbundle(osa, s, sysseq, rays_list, numrays, wavelength):
    '''
    Initialises the Initialbundles
    '''    

    # We want to build a Matrix with the initialbundles as entrys
    # get size of Matrix
    p = len(wavelength)
    q=len(rays_list)
    # initialize Matrix
    initialbundle = [[0 for x in range(p)] for y in range(q)]
    r2 = []

    #Iterate over all entries
    counteri = 0
    counterj = 0
    for bundle in range(q):
        i=rays_list[bundle]["startx"] 
        j=rays_list[bundle]["starty"] 
        k=rays_list[bundle]["startz"] 
        l=rays_list[bundle]["angley"] 
        m=rays_list[bundle]["anglex"] 
        n=rays_list[bundle]["radius"] 
        counterj = 0
        #Setup dict for current Bundle
        bundle_dict = {"startx":i, "starty":j, "startz":k,
                       "angley":l, "anglex":m, "radius":n,
                       "rasterobj":rays_list[bundle]["raster"]}
        for o in wavelength :
            (o1, k1, E1) = osa.collimated_bundle(numrays,
                                        bundle_dict, wave=o)
            initialbundle[counteri][counterj] = \
                 RayBundle(x0=o1, k0=k1, Efield0=E1, wave=o)
            counterj = counterj + 1
        counteri = counteri + 1
    return initialbundle

def get_bundle_merit(osa, s, sysseq, rays_dict, numrays=10,
                     wavelength=[0.587e-3], whichmeritfunc='standard',
                     error='error2', sample_param='wave', penalty=True,
                     penaltyVerz=False,f=100.0):
    """
    initializes the initialBundles and forms the meritfunction
    this is necessary as the meritfunction needs the initalbundle, but in the 
    optimization the initialBundle is not a Uebergabeparameter. Therefore the 
    meritfunction needs to be wrapped inside the bundle initialisation

    f: Brennweite
    """


    initialbundle = buildInitialbundle(osa, s, sysseq, rays_dict, 
                                       numrays, wavelength)
    # determine the number of bundles
    numbundles = len(initialbundle)
    # determine the true number of rays of a bundle
    # in general numrays_true != numrays holds
    numrays_true = initialbundle[0][0].x.shape[2]
    # determine the number of different wavelengths, here the assumption is made
    # that all bundles are created for the same number of different wavelength
    numwaves   = len(initialbundle[0])
    # calculate the number of rays for a bundle for all different wavelengths
    numrays_waves = numwaves*numrays_true

    # --- define meritfunctions:
    # You can add your meritfunctionsrms-implementation and then add it to 
    # the dictionary. If the string whichmeritfunc is not a key in the dict. the 
    # standard_error2 function is taken. (exception handling)
    # This is a necessary generalization, because for instance the sgd needs 
    # a special meritfunc.

    # ---------------standard meritfunction
    def meritfunctionrms_standard(my_s, **kwargs):
        """
        Standard meritfunctionrms: 
        """
        res = 0

        # Loop over all bundles
        for i in range(0, numbundles):
            x = np.array([])
            y = np.array([])
            xChief = np.array([])
            yChief = np.array([])

            # Loop over wavelenghts
            for j in range(0, numwaves):
                my_initialbundle = copy.copy(initialbundle[i][j])
                rpaths = my_s.seqtrace(my_initialbundle, sysseq)

                # put x and y value for ALL wavelengths in x and y array 
                # to caculate mean
                x = np.append(x, rpaths[0].raybundles[-1].x[-1, 0, :])
                y = np.append(y, rpaths[0].raybundles[-1].x[-1, 1, :])

                # penalize overlapping lenses:
                SubInt = len(rpaths[0].raybundles)-4
                diff = np.zeros([numrays_true,SubInt])    
                for w in range(SubInt):
                    for t in range(len(rpaths[0].raybundles[w+3].x[0,2])):
                        if math.isnan(rpaths[0].raybundles[w+3].x[-1,2,t]):
                            pass
                        else:
                            diff[t,w] = rpaths[0].raybundles[w+3].x[-1,2,t] -\
                                        rpaths[0].raybundles[w+3].x[0,2,t]
                #check for overlapping lenses (negative value in diff):
                overlap = 0.0
                for t in range(len(diff)):
                    for w in range(len(diff[0])):
                        if (diff[t,w] < 0.0):
                            overlap += abs(diff[t,w])
                res += (overlap+1.0)**(6) - 1.0          # wie gewichten?
                
                if (j==0):              # Verzeichnung is only referenced to reference wavelength
                    if penaltyVerz:     # is actual only working for collimated bundles
                        # check Verzeichnung: should be smaller than 1%
                        angle = rays_dict[i]["anglex"]
                        SBH = f * math.tan(angle)       # Sollbildhoehe in x Richtung
                        if (len(rpaths[0].raybundles[-1].x[-1,1]) > 0):    # wenn hauptstrahl ueberhaupt auf bild ankommt, dann
                            Y = rpaths[0].raybundles[-1].x[-1,1,0]          # y Komponente des Hauptstrahls auf Bild
                            if Y >= SBH + 0.01*abs(SBH) or Y <= SBH - 0.01*abs(SBH):
                                res += 1000*(abs(Y-SBH)-(0.01*abs(SBH)))**(2)                 # wie gewichten???


                # compute Chief ray of reference wavelength (green one):
                if (j==0):                      # green w.l. should be in the front
                    if (len(rpaths[0].raybundles[-1].x[-1,0])>0):
                        xChief = np.append(xChief, rpaths[0].raybundles[-1].x[-1,0,0])
                    if (len(rpaths[0].raybundles[-1].x[-1,1])>0):            
                        yChief = np.append(yChief, rpaths[0].raybundles[-1].x[-1,1,0])

            # this could be simplified, for the sake of clarity we do not
            if (penalty == True):
                res += (math.exp(numrays_waves/(len(x)+1e-1))-\
                        math.exp(numrays_waves/(numrays_waves+1e-1)))
            # Choose error function
            if (error == 'error2'):
                if (len(xChief)>0 and len(yChief)>0):
                    res += error2squared(x,xChief,y,yChief)
            elif (error == 'error1'):
                if (len(xChief)>0 and len(yChief)>0):
                    res += error1(x,xChief,y,yChief)

        return res

    # ---------------sgd meritfunction
    def meritfunctionrms_sgd(my_s, **kwargs):
        if (len(kwargs) == 0):
            return meritfunctionrms_standard(my_s)
        else:
            res = 0
            # the stochastic gradient function has to tell this function which 
            # number has been drawn, because this function gets invoked more than
            # two times to calculate the gradient and because of that one can not
            # draw the random number within this function
            sample_num = kwargs['sample_num']

            # choose the sampled bundle
            if (sample_param == 'bundle'):
                sample_initialbundle = initialbundle[sample_num]
            if (sample_param == 'wave'):
                sample_initialbundle = [initialbundle[sample_num//numwaves]\
                                                     [sample_num%numwaves]]
            if (sample_param == 'ray'):
                # TODO:
                # this option can cause an error, because the number of rays 
                # numrays is predetermined and therefore can not be changed 
                # during an optimization run but it can happen that not all rays
                # hit the image plane. Because of that a random number can be 
                # drawn but there is no associated ray. the mask, which has then
                # the wrong size will cause an error.
                bundlenum = sample_num//(numwaves*numrays_true)
                wavenum   = (sample_num-numwaves*numrays_true*bundlenum)//\
                            numrays_true
                raynum    = sample_num-numwaves*numrays_true*bundlenum-\
                            numrays_true*wavenum

                sample_initialbundle = [initialbundle[bundlenum][wavenum]]
                mask = np.zeros(numrays_true, dtype=bool)
                mask[raynum] = True

            # Loop over sample_initialbundle
            x = np.array([])
            y = np.array([])
            for i in range(0, len(sample_initialbundle)):

                # calculate the ray paths
                bundle = copy.copy(sample_initialbundle[i])
                rpaths = my_s.seqtrace(bundle, sysseq)

                # append x and y for each bundle
                x = np.append(x, rpaths[0].raybundles[-1].x[-1, 0, :])
                y = np.append(y, rpaths[0].raybundles[-1].x[-1, 1, :])
 
                # Add up all the mean values of the different wavelengths
                xmean = np.mean(x)
                ymean = np.mean(y)

                # if the sample parameter is ray only one ray has been drawn
                if (sample_param == 'ray'):
                    x = x[mask]
                    y = y[mask]

                # this could be simplified, for the sake of clarity we do not
                if (penalty == True):
                    res += (math.exp(numrays_true/(len(x)+1e-1))-\
                            math.exp(numrays_true/(numrays_true+1e-1)))

                # Choose error function
                if (error == 'error2'):
                    res += error2squared(x, xmean, y, ymean)
                elif (error == 'error1'):
                    res += error1(x, xmean, y, ymean)

            return res

    # for defining which meritfunction should be used
    switcher = {'standard': meritfunctionrms_standard,
                'sgd'     : meritfunctionrms_sgd      }

    return (initialbundle, switcher.get(whichmeritfunc, \
                                        meritfunctionrms_standard))

