import numpy as np
import math
from pyrateoptics.optimize.optimize         import Optimizer
import time
from project_optimize_backends              import ProjectScipyBackend
from copy import deepcopy
import scipy


def benchmark(OpticalSystem,meritfunctionrms,updatefunction,methods):
    """
    methods: 2D-Liste der Form: methods = [[name,methodparam1,methodparam2,...],...]
        i.e. methods = [["Powell","penalty","penalty-lagrange"],["Nelder-Mead","penalty"]]
        Achtung: methods out of scipy need the " at the beginning and the end -> ["Powell","penalty",...
        ouer own methods doesnt need this -> e.g. [test_minimize_neldermead,"penalty",...
    meritfunction = meritfunction you want to use
    updatefunction = updatefunction you want to use

    For NlOpt you have to do:
                pip install nlopt --no-clean
    """
    # All for Scipy-Methods:----------------------------------------------------   """
    # Create a list with ProjectScipyBackend-Objects for all Methods:
    listOptBackend = []

    for i in range(len(methods)):
        for j in range(len(methods[i])-1):
            listOptBackend.append(ProjectScipyBackend(optimize_func=methods[i][0], \
                                  methodparam=methods[i][j+1], \
                                  options={'maxiter':100, 'xatol':1e-10, \
                                           'fatol':1e-10, 'gtol':1e-10, \
                                           'disp':True}))

    # create a list to safe results:
    header = ["Verfahren", "Pen/Lag", "MeritFinal", "Time [s]", "fEval","Iter", "xFinal", "Bound ok?"]
    Result = [[] for i in range(len(listOptBackend)+1)]
    Result[0].append(header)
    for i in range(len(listOptBackend)):
        Result[i+1].append(listOptBackend[i].optimize_func)
        if (listOptBackend[i].methodparam == "standard"):
            Result[i+1].append("None")
        elif  (listOptBackend[i].methodparam == "penalty"):
            Result[i+1].append("Penalty")
        elif (listOptBackend[i].methodparam == "penalty-lagrange"):
            Result[i+1].append("Pen + Lag")   
        elif (listOptBackend[i].methodparam == "log"):
            Result[i+1].append("Barrier-Met.")   
 

    # loop over all methods:
    counter1 = 1
    for backend in listOptBackend:
        OS = deepcopy(OpticalSystem)            # copy of OS for every method
        optimi = Optimizer(OS,
                    meritfunctionrms,
                    backend=backend,
                    updatefunction=updatefunction)

        # Update constraints:
        backend.update_PSB(optimi)
        
        # is done in upadate_PSB
        """
        # Update parameter "bounds":
        # needed for scipy-algos: "L-BFGS-B","TNC","SLSQP","trust-constr"
        lb = np.empty([len(backend.bdry)/2])
        ub = np.empty([len(backend.bdry)/2])

        for i in range (len(backend.bdry)/2):
            lb[i] = backend.bdry[2*i]
            ub[i] = backend.bdry[2*i+1]
        bound = scipy.optimize.Bounds(lb,ub)
        backend.kwargs['bounds'] = bound
        """
        bdry = backend.kwargs['bounds']

        # Update Jacobi-Matrix for case methodparam == 1:
        # (the other cases are treated in "project_minimize_backends")
        def jacobian(x):
            h = np.sqrt(np.finfo(float).eps)
            func = optimi.MeritFunctionWrapper
            grad = np.empty([len(x)])
            E = np.eye(len(x))
            for i in range (len(x)):
                grad[i] = (func(x+h*E[i,:])-func(x-h*E[i,:])) / (2*h)
            return grad

        #backend.kwargs['jac'] = jacobian

        """    
        # Functions for jacobian and hessian for scipy.minimize function:
        def jacobian(x):
            eps = np.sqrt(np.finfo(float).eps)
            func=optimi.MeritFunctionWrapper
            grad = np.ones(len(x))
            for i in range(len(x)):
                x_neu1 = deepcopy(x)
                x_neu2 = deepcopy(x)
                x_neu1[i] = x_neu1[i] + eps
                x_neu2[i] = x_neu2[i] - eps
                grad[i] = (func(x_neu1)-func(x_neu2))/(2*eps)
            return grad

        def hessian(x):
            eps = np.sqrt(np.finfo(float).eps)
            func = optimi.MeritFunctionWrapper
            hes = np.ones((len(x),len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    x_neu1 = deepcopy(x)
                    x_neu2 = deepcopy(x)
                    x_neu3 = deepcopy(x)
                    x_neu1[i] += eps
                    x_neu1[j] += eps
                    x_neu2[i] += eps
                    x_neu3[j] += eps

                    hes[i,j] = (func(x_neu1)-func(x_neu2)-func(x_neu3)+func(x)) / eps**2
            return hes
        
            #temp = {"jac": jacobian, "hess": hessian}   # add to options of backend
            #met.options.update(temp)
        """


        # start optimization run and messure the time:
        start = time.time()
        OS = optimi.run()
        ende = time.time()

        # save results in list "Result":
        Result[counter1].append(backend.res.fun)        # save final merit
        Result[counter1].append(ende-start)             # save time
        Result[counter1].append(optimi.NoC)             # number of calls of merit func
        if (backend.methodparam == "standard"):         # number Iterations
            Result[counter1].append(backend.res.nit)    # actual i do not know the nit by penalty and lagrange -> set to -1
        else:
            Result[counter1].append(-1)
        Result[counter1].append(backend.res.x)          # final x-value

        # check if xfinal fulfills boundaries:
        check = 0
        for k in range (len(backend.res.x)):
            if (backend.res.x[k] >= bdry.lb[k] and backend.res.x[k] <= bdry.ub[k]):
                check += 0
            else:
                check += 1
        if (check >= 1):
            Result[counter1].append("no")
        else:
            Result[counter1].append("yes")
        
        # Increment counter
        counter1 += 1

    # print Result:
    print("====================================================================================================================================================")
    print("Benchmark:")
    print("Bei den penalty-Methoden kenne ich die Gesamte Iterationszahl nicht -> hier ist iter aktuell auf -1")
    print("")
    #print header
    print("%60s %10s %10s %10s %10s %10s %10s %10s" % (header[0],header[1], \
                                                       header[2],header[3], \
                                        header[4],header[5],header[6],header[7]))
    print("     -----------------------------------------------------------------------------------------------------------------------------------------------")
    for i in range(len(Result)-1):
        # print entries for every backend
        print("%60s %10s %10.5f %10.3f %10d %10d %10.3f %10s\n"  % (Result[i+1][0],Result[i+1][1],Result[i+1][2],Result[i+1][3],Result[i+1][4],Result[i+1][5], \
                                                                     Result[i+1][6][0], \
                                                                     Result[i+1][7]))
        # loop over all entries of res.x (without res.x[0], cause already printed above)
        for j in range(len(backend.res.x)-1):
            print("%126.3f\n" % Result[i+1][6][j+1])
    print("====================================================================================================================================================")

 
