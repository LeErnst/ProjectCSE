# aux functions 
from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

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

#---------------------------------------
#Writes data to a file
#---------------------------------------
from pyrateoptics.core.iterators import OptimizableVariableKeyIterator
from pyrateoptics.raytracer.material.material_glasscat import\
     refractiveindex_dot_info_glasscatalog, CatalogMaterial
import matplotlib.pyplot as plt 
import os
import sys

'''
This class is for storing and printing the optimization results
Have a look at the readme file if you want more informations how 
the input/output data can be determined
'''
#---------------------------------------
class inout:
    def __init__(self):
        self.globpath="./Input_Output/"
        fileobj = open(self.globpath+"file_data", "r")
        readdata=[]
        for line in fileobj:
            readdata.append(line.rstrip())
        fileobj.close()
        if(len(readdata[0])==0):
            self.name="result"
        else:
            self.name=readdata[0]
        self.path=readdata[1]
        self.mode=1
        self.filter_status=readdata[2]
        self.iteration=-1
        self.search_result_var={}
        self.search_result_fix={}
        self.shorttolong_var={}
        self.shorttolong_fix={}
        self.longnames_var=[]
        self.longnames_fix=[]
        self.shortnames_var=[]
        self.shortnames_fix=[]
        #Take care if you change the input_surface file. The order below
        #hast to represent the order of the input_surface file
        self.surface_order=["name", "decx", "decxvar", "decy", \
                "decyvar", "decz", "deczvar", "tiltx", "tiltxvar",\
                "tilty", "tiltyvar", "tiltz", "tiltzvar", "maxrad",\
                "maxradvar","minrad", "minradvar", "conic", "conicvar",\
                "curvature", "curvaturevar", "conn1", "conn2", \
                "isstop","shape","aperture"]
        #the name of the stateattribute of the variables. Take care! They
        #have to end like "variablename"var with a var at the end
        self.NamesOptVariable=["decxvar","decyvar","deczvar","tiltxvar",\
                "tiltyvar","tiltzvar","conicvar","curvaturevar"]
        #Take care if you change the bundle_input file. The order below
        #hast to represent the order of the bundle_input file
        self.ray_order=["startx","starty","startz",\
                "radius","anglex","angley",\
                "numrays","wavelength","raster"]
        self.surfaces=np.genfromtxt(self.globpath+"surface_input",dtype=None,\
                comments="#")
        self.rays=np.genfromtxt(self.globpath+"bundle_input",dtype=None,\
                comments="#").tolist()
        self.material=[]
        self.SurfNameList=[]
        #take care: it is necessaray that here stands the "normal" variable name 
        #added with an var. all possible varialbe items have to mentioned below 

    def set_add(self):
        self.mode=0

    def set_first(self):
        delf.mode=1

    def print_data(self):
        print("SAVED DICTIONARY WITH DATA:")
        for (key, item) in self.search_result_item.items():
            print(key + "    " + item)

    def find_char(self, string, character):
        return [i for i, ltr in enumerate(string) if ltr==character]

    def make_shortnames(self):
        for key in self.search_result_var.keys():
            index=self.find_char(key,".")
            short=key[index[-2]+1:]
            self.shorttolong_var[short]=key
            self.shortnames_var.append(short)
    
    #prints the shortnames and the associated real (long) names
    def print_shortnames(self):
        print("SHORTNAMES ARE")
        print("^^^^^^^^^^^^^^^^")
        for (shortn, longn) in self.shorttolong_var.items():
            print(shortn +"      :=      " + longn)
    
    #argv are the shortnames that sould be printed->with this function it is 
    #possible to chosse the variables that should be printed
    #TODO Didnt work yet. All variable variables are plotted at the moment
    def plot_data(self,*argv):
        #fig1=plt.figure()
        plt.figure()
        plt.xlabel("iterations")
        plt.ylabel("values")
        for i in range(0,len(self.search_result_var.items())):
            plt.plot(range(self.iteration+1),\
                    self.search_result_var[self.longnames_var[i]])
        self.make_shortnames()
        plt.legend(self.shortnames_var)
        #fig1.show()
        if(len(self.path)==0):
            plt.savefig(self.name + ".png")
        else:
            plt.savefig(self.path+self.name+".png")
        #plt.show()

    def write_to_file(self):    
        if(self.filter_status=="NONE"):
            print("No output file generated. Specify this if needed in the file_data file")
        else:
            if(len(self.path)==0):
                dir1=self.name+"_fixed"+".csv"
                dir2=self.name+"_variable"+".csv"
            else:
                dir1=self.path+self.name+"_fixed"".csv"
                dir2=self.path+self.name+"_variable"".csv"

            if(self.filter_status=="VARIABLE" or \
                    self.filter_status=="VARIABLEANDFIXED"):
                with open(dir2, 'w') as f:
                    for key in self.search_result_var.keys():
                        f.write("%s,%s\n"%(key,self.search_result_var[key]))
                f.close()    
            if(self.filter_status=="FIXED" or \
                    self.filter_status=="VARIABLEANDFIXED"): 
                with open(dir1, 'w') as f:
                    for key in self.search_result_fix.keys():
                        f.write("%s,%s\n"%(key,self.search_result_fix[key]))
                f.close()
            #if you added residuum or merritfunction value etc. here it hast to be written to the code

    def store_data(self,os):
        lst = OptimizableVariableKeyIterator(os).variables_dictionary
        for (key,obj) in lst.items():
            if(self.filter_status=="VARIABLE" or \
                    self.filter_status=="VARIABLEANDFIXED"):
                if(obj.var_type()=="variable"):
                    if(self.mode==1):
                        self.longnames_var.append(key)
                        self.search_result_var[str(key)]=[obj.evaluate()]
                    else:
                        self.search_result_var[str(key)].append(obj.evaluate())
            if((self.filter_status=="FIXED" \
                    or self.filter_status=="VARIABLEANDFIXED") and self.mode==1):
                if(obj.var_type()=="fixed"):
                    if(self.mode==1):
                        self.longnames_fix.append(key)
                        self.search_result_fix[str(key)]=[obj.evaluate()]
                    else:
                        self.search_result_fix[str(key)].append(obj.evaluate())
        if(self.mode==1):
            self.set_add()
        
        self.iteration+=1
        #print(self.search_result_fix)
        #print(self.search_result_var)

        #if the residuum or merritfunction value is needed it can be added here in the code 
    
    def get_sufval(self, name, index):
        return self.surfaces[index][self.surface_order.index(name)]

    def get_rayval(self,name):
        return self.rays[self.ray_order.index(name)]

    #this function creates for every surface (defined in the surface_input file) one coordinate system, and returns a list of them
    def create_coordinate_systems(self,optical_system):
        coordinate_systems=[]
        for i in range(self.surfaces.shape[0]):
            if(i==0):
                refname_=optical_system.rootcoordinatesystem.name
            else:
                refname_=coordinate_systems[i-1].name
            coordinate_systems.append(optical_system.\
                    addLocalCoordinateSystem(LocalCoordinates.p(\
                    name=self.get_sufval("name",i),\
                    decx=self.get_sufval("decx",i), decy=self.get_sufval("decy",i), \
                    decz=self.get_sufval("decz",i), tiltx=self.get_sufval("tiltx",i),\
                    tilty=self.get_sufval("tilty",i), tiltz=self.get_sufval("tiltz",i)),\
                    refname=refname_))

        return coordinate_systems

    #this function creates the sufaces defined in surface_input file and returns a list of these surfaces 
    def create_surfaces(self, cs):  #cs=coordinate Systems
        surface_objects=[]
        for i in range(self.surfaces.shape[0]):
            if(self.get_sufval("shape",i)=="Conic"):
                #hier noch cc einfuegen
                shape_=Conic.p(cs[i],cc=float(self.get_sufval("conic",i)),curv=float(self.get_sufval("curvature",i)))
                #print(self.get_sufval("curvature",i))
            else:
                shape_=None
            if(self.get_sufval("aperture",i)=="Circular"):
                aperture_=CircularAperture(cs[i],minradius=float(self.get_sufval("minrad",i)), maxradius=float(self.get_sufval("maxrad",i)))
                #print(float(self.get_sufval("maxrad",i)).__class__.__name__)
            else:
                aperture_=None

            self.SurfNameList.append(self.get_sufval("name",i))
            surface_objects.append(Surface.p(cs[i], shape=shape_, aperture=aperture_))

        return surface_objects

    #reads material from the input_surface file and creates the needed object
    #take care! A database of all materials is necessary. 
    #Store the database in refractiveindex.info-database. Different databases
    #can be downloaded from github e.g.
    #https://github.com/polyanskiy/refractiveindex.info-database
    def create_material(self, cs, elem1, surf):
        if os.path.isdir("../pyrateoptics/refractiveindex.info-database/database/"):
            for i in range(self.surfaces.shape[0]):
                tempmat1=self.get_sufval("conn1",i)
                tempmat2=self.get_sufval("conn2",i)
                if(self.material.count(tempmat1)==0 and tempmat1 != "None"):
                    self.material.append(tempmat1)
                if(self.material.count(tempmat2)==0 and tempmat2 != "None"):
                    self.material.append(tempmat2)
            
            gcat = refractiveindex_dot_info_glasscatalog("../pyrateoptics/refractiveindex.info-database/database/")
            for i in range(len(self.material)):
                #absolutely no idea why there is a coordinate system necessary for creating a material. random choice: use always cs[0].
                #result is always the same, no matter which coordinate system is used
                tempmat=gcat.createGlassObjectFromLongName(cs[0],self.material[i])
                elem1.addMaterial(self.material[i],tempmat)
             
            for i in range(self.surfaces.shape[0]):
                elem1.addSurface(self.get_sufval("name",i),surf[i], \
                        (self.get_sufval("conn1",i),self.get_sufval("conn2",i)))
        else:
            print("It looks like there is no database at refractiveindex.info-database")
            print("Please download a database e.g. from https://github.com/polyanskiy/refractiveindex.info-database")

    #"value1&value2&value3" -> [value1,value2,value3] all values get a float 
    #cast. If the cast is not possible the input string is returned 
    def convert_2_list(self,string):
        try:
            liste=list(string.split("&"))                                                    
            newlist=[]
            for i in range(len(liste)):
                newlist.append(float(liste[i]))
            return newlist
        except:
            return string

    #creates optimizable variables with fixed or variable state
    #if in the surface_input file are intervals defined, the variables
    #get modified in the determined way
    def setup_variables(self, os, elemName):
        optiVarsDict = {}                                                                
        for i in range(len(self.SurfNameList)):
            temptdict={}
            for item in self.NamesOptVariable:
                if self.get_sufval(item,i)!="f":
                    temptdict[item[:-3]]=self.convert_2_list(self.get_sufval(item,i))
            
            if(len(temptdict)!=0): optiVarsDict[self.get_sufval("name",i)]=temptdict


        for surfnames in optiVarsDict.keys():
            for params in self.NamesOptVariable[:-2]:
                if params[:-3] in optiVarsDict[surfnames]:
                    decOrTilt = getattr(os.elements[elemName].surfaces[surfnames].\
                            rootcoordinatesystem, params[:-3])
                    decOrTilt.toVariable()
                    decOrTilt.setInterval(left=optiVarsDict[surfnames][params[:-3]][0],
                            right=optiVarsDict[surfnames][params[:-3]][1])

            for params in self.NamesOptVariable[-2:]:
                if params[:-3] in optiVarsDict[surfnames]:
                    curvOrCc = getattr(os.elements[elemName].\
                            surfaces[surfnames].shape, params[:-3])
                    curvOrCc.toVariable()
                    curvOrCc.setInterval(left=optiVarsDict[surfnames]\
                            [params[:-3]][0], \
                            right=optiVarsDict[surfnames][params[:-3]][1])

    #returns a sequence of surfaces which are defined in the surface_input file 
    def get_sysseq(self, elem1):
        templist=[]
        for i in range(len(self.SurfNameList)):
            if self.get_sufval("isstop",i):
                templist.append((self.get_sufval("name",i), {"is_stop":True}))
            else:
                templist.append((self.get_sufval("name",i), {}))
            
        sysseq=[(elem1.name,templist)]
        return sysseq 

    #returns the dictionary which represents the ray bundles 
    def get_rays_dict(self):
        rays_dict={}
        for i in range(len(self.ray_order)):
            value=self.get_rayval(self.ray_order[i])
            name=self.ray_order[i]
            if isinstance(value,int):
                rays_dict[name]=[value]
            elif name=="raster":
                if(value=="RectGrid"):
                    rays_dict[name]=raster.RectGrid()
            else:    
                rays_dict[self.ray_order[i]]=self.convert_2_list(value)
        return rays_dict 

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def error2squared(x, x_ref, y, y_ref, penalty=False):
    '''
    computes the squared
    '''
    if penalty:
        res = np.sum((x - x_ref)**2 + (y - y_ref)**2) + 10.*math.exp(-len(x))
    else:
        res = np.sum((x - x_ref)**2 + (y - y_ref)**2) 

    return res


def error1(x, x_ref, y, y_ref, penalty=False):
    '''
    computes the 
    L1-error = sum_{i=1 to #rays}(||(x_i, y_i)^T - (x_ref, y_ref)^T||_1)
    '''
    if penalty:
        res = np.sum(np.absolute(x - x_ref) + np.absolute(y - y_ref))\
              + 10.*math.exp(-len(x))
    else:
        res = np.sum(np.absolute(x - x_ref) + np.absolute(y - y_ref))

    return res


def setOptimizableVariables(os, elemName, optiVarsDict, SurfNamesList):
    '''
    os: OpticalSystem object
    optiVarsDict: dictionary for all parameters
    SurfNamesList: dictionary with all surface names
    elemName: name of the optical element

    transforms all the parameters in optiVarsList into the variable state
    '''

    allParamsList = ["decz", "decx", "decy", "tiltx", "tilty", "tiltz",
                     "curvature", "cc"]

    for surfnames in optiVarsDict.keys():
        for params in allParamsList[:-2]:
            if params in optiVarsDict[surfnames]:
                decOrTilt = getattr(os.elements[elemName].surfaces[surfnames].\
                                    rootcoordinatesystem, params)
                decOrTilt.toVariable()
                decOrTilt.setInterval(left=optiVarsDict[surfnames][params][0],
                                      right=optiVarsDict[surfnames][params][1])

        for params in allParamsList[-2:]:
            if params in optiVarsDict[surfnames]:
                curvOrCc = getattr(os.elements[elemName].surfaces[surfnames].\
                                   shape, params)
                curvOrCc.toVariable()
                curvOrCc.setInterval(left=optiVarsDict[surfnames][params][0],
                                     right=optiVarsDict[surfnames][params][1])


def calcBundleProps(osa, bundleDict, numrays_plot=100):
    '''
    osa: OpticalSystemAnalysis-object
    bundleDict: all bundle data in a dictionary
    numrays_plot: number of rays for plotting
    
    return are the two dictionaries with all o1,k1,E0 matrizes
    '''
    bundlePropDict      = {}
    bundlePropDict_plot = {}

    for i in bundleDict.keys():
        bundlePropDict[i] = getattr(osa, bundleDict[i][3])(bundleDict[i][0],
                                                           bundleDict[i][1],
                                                           bundleDict[i][2])

    for i in bundleDict.keys():
        bundlePropDict_plot[i] = getattr(osa, bundleDict[i][3])(numrays_plot,
                                                                bundleDict[i][1],
                                                                bundleDict[i][2])
    return bundlePropDict, bundlePropDict_plot


def calculateRayPaths(os, bundleDict, bundlePropDict, sysseq):
    '''
    os: OpticalSystem
    bundleDict: all bundle data in a dictionary
    bundlePropDict: dictionary which contains all the bundle properties(o1,k1,E1)
                    from above
    sysseq: system sequence

    return are the x,y vectors which contain the (x,y)-coordinates from the 
    raytracer 
    '''
    x = np.array([])
    y = np.array([])

    for i in bundlePropDict.keys():
        bundle = RayBundle(x0      = bundlePropDict[i][0], 
                           k0      = bundlePropDict[i][1],
                           Efield0 = bundlePropDict[i][2],
                           wave    = bundleDict[i][2])
        rpaths = os.seqtrace(bundle, sysseq)
        x = np.append(x, rpaths[0].raybundles[-1].x[-1, 0, :])
        y = np.append(y, rpaths[0].raybundles[-1].x[-1, 1, :])
    
    return x, y

def plotBundles(s, initialbundle, sysseq, 
                ax, pn, up2, color="blue"):
    '''
    EDITED BY LEANDRO 29.11.2019
    os: OpticalSystem-object optimization
    bundleDict: all bundle data in a dictionary
    bundlePropDict_plot: dictionary which contains all the bundle properties
                         (o1,k1,E1) for plotting
    sysseq: system sequence
    color: the color for the rays

    draws all rays in the bundleDict and the optical system os
    '''

    # Get dimensions and initialise r2
    m = len(initialbundle)
    n = len(initialbundle[0])
    r2 = [0 for x in range(m*n)]

    # Calculate rays
    counter = 0
    for i in range(0,m):
        for j in range(0,n):
            r2 = s.seqtrace(initialbundle[i][j], sysseq)
            for r in r2:
                r.draw2d(ax, color="green", plane_normal=pn, up=up2)

    s.draw2d(ax, color="grey", vertices=50, plane_normal=pn, up=up2) 

def plotSpotDia(osa, numrays, rays_dict, wavelength):

    #Set defaults for dictionary
    rays_dict.setdefault("startx", [0])
    rays_dict.setdefault("starty", [0])
    rays_dict.setdefault("startz", [-7])
    rays_dict.setdefault("angley", [0])
    rays_dict.setdefault("anglex", [0])
    rays_dict.setdefault("rasterobj", raster.RectGrid())
    rays_dict.setdefault("radius", [15])

    #Iterate over all entries
    for i in rays_dict["startx"] :
        for j in rays_dict["starty"] :
            for k in rays_dict["startz"] :
                for l in rays_dict["angley"] :
                    for m in rays_dict["anglex"] :
                        for n in rays_dict["radius"] :
                            #Setup dict for current Bundle
                            bundle_dict = {"startx":i, "starty":j, "startz":k,
                                           "angley":l, "anglex":m, "radius":n,
                                           "rasterobj":rays_dict["rasterobj"]}
                            for o in wavelength :
                                osa.aim(numrays, bundle_dict, wave=o)
                                osa.drawSpotDiagram()


def get_bdry(optimi) :
    '''
    Returns an array that has the boundaries of the optimizable
    variable in it, meaning:
    [lowerbound_x1 upperbound_x1 ... lowerbound_xn upperbound_xn]
    '''
    
    n = len(optimi.collector.variables_list)
    bdry = np.zeros(2*n)
    cnt = 0
    for i in optimi.collector.variables_list :
        bdry[cnt] = i.interval_trafo_fo._FunctionObject__global_variables['left']
        cnt = cnt + 1
        bdry[cnt] = i.interval_trafo_fo._FunctionObject__global_variables['right']
        cnt = cnt + 1

    return bdry


def eval_h(x, bdry) :
    '''
    returns vector with the values of h(x) according to
    Nocedal p. 494, meaning evaluating the min(0,h(x_i))
    respectively max(0,h(x_i)) (depending whether it is
    upper/lower boundary
    '''

    x = np.asfarray(x).flatten()
    h_x = np.zeros(len(bdry))
    cnt = 0
    for xi in x :
        h_x[cnt] = np.maximum(0, bdry[cnt]-xi)  #lower bound
        cnt = cnt + 1
        h_x[cnt] = np.maximum(0, xi-bdry[cnt])  #upper bound
        cnt = cnt + 1

    return h_x


def eval_c(x, bdry) :
    '''
    returns vector with the values of c(x), with c(x) being the function
    c(x) >= 0
    '''

    x = np.asfarray(x).flatten()
    c_x = np.zeros(len(bdry))
    cnt = 0
    for xi in x :
        c_x[cnt] = xi - bdry[cnt] #lower bound
        cnt = cnt + 1
        c_x[cnt] = bdry[cnt] - xi  #upper bound
        cnt = cnt + 1

    return c_x


def my_log(x) :
    '''
    Returns log(x) if x>0, and -999999999999999 if x<=0
    '''

    res = np.zeros(len(x))
    cnt = 0
    for xi in x :
        if xi>0:
            res[cnt] = np.log(xi)
            cnt = cnt + 1
        else:
            res[cnt] = -9999999999999999
            cnt = cnt + 1

    return res

def printArray(name, x, typ='float', point=3):                                   
    '''
    point is the number of digits after the point
    '''
    x = np.asarray(x)

    if (typ == 'float'):
        style = '{:0.'+str(point)+'f}'
        np.set_printoptions(formatter={typ: style.format})
    elif (typ == 'int'):
        x = np.array(x, dtype=int)
        np.set_printoptions(formatter={typ: '{:d}'.format})
    elif (typ == 'bool'):
        np.set_printoptions(formatter={typ: '{:b}'.format})
    else:
        style = '{:0.3f}'
        np.set_printoptions(formatter={typ: style.format})
    print('\n', end='')
    print(name, end=' ')
    print(x, end='')
    print('\n')


def termcondition_descdir(gradient, gradtol=1e-8):
    dim = len(gradient)
    E = np.eye(dim,dim)
    ismin = True
    for i in range(dim):
        if (ismin):
            if (np.dot(E[i,:],gradient) < 0):
                ismin = False
            if (np.dot(-E[i,:],gradient) < 0):
                ismin = False

#    if (np.linalg.norm(gradient, np.inf) >= gradtol):
#        ismin = False

    return ismin


