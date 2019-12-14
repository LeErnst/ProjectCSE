# --- general
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
from pyrateoptics.raytracer.analysis.optical_system_analysis import OpticalSystemAnalysis
from pyrateoptics.raytracer.analysis.surface_shape_analysis  import ShapeAnalysis
# --- optimization
from pyrateoptics.optimize.optimize          import Optimizer
from pyrateoptics.optimize.optimize_backends import (ScipyBackend,
                                                     Newton1DBackend,
                                                     ParticleSwarmBackend,
                                                     SimulatedAnnealingBackend)

# --- auxiliarys
from auxiliary_functions import error2squared, error1


def buildInitialbundle(osa, s, sysseq, rays_dict, numrays=10, wavelength=[0.587e-3]):
    '''
    Initialises the Initialbundles
    '''    
    #Set defaults for dictionary
    rays_dict.setdefault("startx", [0])
    rays_dict.setdefault("starty", [0])
    rays_dict.setdefault("startz", [-7])
    rays_dict.setdefault("angley", [0])
    rays_dict.setdefault("anglex", [0])
    rays_dict.setdefault("rasterobj", raster.RectGrid())
    rays_dict.setdefault("radius", [15])
    
    # We want to build a Matrix with the initialbundles as entrys
    # get size of Matrix
    p = len(wavelength)
    q = len(rays_dict["startx"])*len(rays_dict["starty"])*len(rays_dict["startz"])*\
    len(rays_dict["angley"])*len(rays_dict["anglex"])*len(rays_dict["radius"])
    # initialize Matrix
    initialbundle = [[0 for x in range(p)] for y in range(q)]
    r2 = []

    #Iterate over all entries
    counteri = 0
    counterj = 0
    for i in rays_dict["startx"] :
        for j in rays_dict["starty"] :
            for k in rays_dict["startz"] :
                for l in rays_dict["angley"] :
                    for m in rays_dict["anglex"] :
                        for n in rays_dict["radius"] :
                            counterj = 0
                            #Setup dict for current Bundle
                            bundle_dict = {"startx":i, "starty":j, "startz":k,
                                           "angley":l, "anglex":m, "radius":n,
                                           "rasterobj":rays_dict["rasterobj"]}
                            for o in wavelength :
                                (o1, k1, E1) = osa.collimated_bundle(numrays,
                                                            bundle_dict, wave=o)
                                initialbundle[counteri][counterj] = RayBundle(x0=o1, k0=k1, Efield0=E1, wave=o)
                                counterj = counterj + 1
                            counteri = counteri + 1
    return initialbundle

def get_bundle_merit(osa, s, sysseq, rays_dict, numrays=10, wavelength=[0.587e-3]):
    """
    initializes the initialBundles and forms the meritfunction
    this is necessary as the meritfunction needs the initalbundle, but in the 
    optimization the initialBundle is not a Uebergabeparameter. Therefore the 
    meritfunction needs to be wrapped inside the bundle initialisation
    """

    initialbundle = buildInitialbundle(osa, s, sysseq, rays_dict, numrays, wavelength)
    # --- define meritfunctioni
    def meritfunctionrms(my_s):
        """
        Merit function for tracing a raybundle through system and calculate
        rms spot radius without centroid subtraction. Punish low number of
        rays, too.
        """
        res = 0

        # Loop over all bundles
        for i in range(0, len(initialbundle)):
            x = []
            y = []
            # Loop over wavelenghts
            for j in range(0, len(initialbundle[0])):
                my_initialbundle = initialbundle[i][j]
                rpaths = my_s.seqtrace(my_initialbundle, sysseq)

                # put x and y value for ALL wavelengths in x and y array to caculate mean
                x.append(rpaths[0].raybundles[-1].x[-1, 0, :])
                y.append(rpaths[0].raybundles[-1].x[-1, 1, :])

            # Add up all the mean values of the different wavelengths
            xmean = np.mean(x)
            ymean = np.mean(y)

        # Choose error function
        res += error2squared(x, xmean, y, ymean)
        # res += error1(x, x_ref, y, y_ref)
        # res = res + np.sum((x - xmean)**2 + (y - ymean)**2) + 10.*math.exp(-len(x))
        return res
    
    return (initialbundle, meritfunctionrms)
