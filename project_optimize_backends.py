from __future__ import print_function
import math
import random
import numpy
import scipy
import sys
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from pyrateoptics.core.log import BaseLogger
from pyrateoptics.optimize.optimize_backends import Backend
from derivatives import grad, grad_pen, grad_lag, grad_log, hessian
from auxiliary_functions import get_bdry, eval_h, eval_c, my_log, printArray,\
                                termcondition, plot2d, plot3d,\
                                line_search_bound

class ProjectScipyBackend(Backend):
    def __init__(self, optimize_func, methodparam=None, tau0=2.0,
                 options={}, stochagradparam=False, **kwargs):
        self.optimize_func   = optimize_func
        self.options         = options
        self.methodparam     = methodparam
        self.stochagradparam = stochagradparam
        self.kwargs          = kwargs
        self.tauk            = tau0
        self.lamk            = 1.
        self.stochagrad      = None

        #set tolerance for gradient to check if x is a minima
        if 'gtol' in options.keys() :
            self.gtol = options.get('gtol')
        else :
            self.gtol = 1e-8

        # self.func = MeritFunctionWrapper is set within the Optimizer __init__ 

    def update_PSB(self, optimi) :
        '''
        function gets the boundaries for the variables. THis is necessary if 
        you want to run the optimization with penalty/lagrange terms. It is
        important to call this function AFTER the Optimizer object was created
        and BEFORE you run the optimization.
        '''
        # calculate and assign the boundaries
        self.bdry = get_bdry(optimi)
        printArray('bdry in ProjectScipyBackend = ', self.bdry)

        # make the boundaries available for the methods
        # get the boundaries for the scipy algos
        lb = numpy.empty([len(self.bdry)/2])
        ub = numpy.empty([len(self.bdry)/2])
        for i in range(len(self.bdry)/2):
            lb[i] = self.bdry[2*i]
            ub[i] = self.bdry[2*i+1]
        # make it available 
        self.kwargs['bounds'] = scipy.optimize.Bounds(lb,ub)
        
    
    def run(self, x0):
        # tolerance for the infeasibility measurement 
        tol_seq = 10*numpy.finfo(float).eps 
        # number of iterations
        iterNum = 0
        # h for the gradient
        h = 1e-8
        # number of digits after the point of xk
        points = 7
        
        printArray('x0 =', x0, point=points)
        print('\nmeritfunction(x_0) = %10.6f' % (self.func(x0)))

        if (self.methodparam == 'standard'):

            # optimize: meritfunction(x)

            print('----------------- run standard -----------------')

            # find local minimizer of meritfunction(x)
            res = minimize(self.func, 
                           x0=x0, 
                           args=(), 
                           method=self.optimize_func,
                           options=self.options, 
                           **self.kwargs)
            #TEST LEANDRO
            #attempt to use numpy, global, stochastic algorithms
            #does not work yet
            #bound =  [[0]*2 for i in range(len(x0))]
            #for i in range(len(x0)/2) :
            #    bound[i][0] = self.bdry[2*i]
            #    bound[i][1] = self.bdry[2*i+1]
                    
            #res = differential_evolution(self.func, bound, maxiter=1, popsize=750, strategy='rand2exp')

        elif (self.methodparam == 'penalty'):

            # optimize: meritfunction(x) + 0.5*tau*||h(x)||_2^2

            print('----------------- run penalty -----------------')

            xk   = x0
            while (1): 
                # define the gradient for the penalty method
                if (self.stochagradparam == True):
                    def stochagrad_total(x):
                        res = self.stochagrad(self.func, x, h) + \
                              grad_pen(x,self.bdry,self.tauk)
                        return res
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_pen(x,self.bdry,self.tauk)
                        return res
                    self.options['stochagrad'] = stochagrad_total
                else: # this is the default case
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_pen(x,self.bdry,self.tauk)
                        return res
                # store the grad-function in options
                self.options['grad'] = grad_total
                
                # update iteration number
                iterNum += 1
                print('\niteration number = %d' % (iterNum))
                printArray('bdry =', self.bdry)
                print('\ntau = %6.3f' % (self.tauk))
                printArray('gradient of the penalty term =',\
                           grad_pen(xk,self.bdry, self.tauk))

                # find local minimizer of 
                # meritfunction(x) + 0.5*tau_k*||h(x)||_2^2
                penalty_func = lambda x: self.func(x) +\
                                         0.5*self.tauk*numpy.square(\
                                         numpy.linalg.norm(eval_h(x,self.bdry)))
                
                res = minimize(penalty_func,
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               jac=grad_total,
                               options=self.options, 
                               **self.kwargs)
                # update xk
                xk = res.x

                printArray('x_k =', res.x, point=points)
                print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))

                # check if xk is in the feasible set with ||h(x)||_inf < 10*eps
                if (iterNum>=5):
                    print('\n----------- end of penalty run -----------')
                    print('\nReached max. numbers of penalty iterations = %d' % \
                          (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))<self.gtol) and 
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('\n----------- end of penalty run -----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))>self.gtol) and
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('Algorithm did not reach a Minimum but the solution is still in the feasible area!')
                    print('Keep Tau and continue with Algorithm')
                else: # xk is not in the feasible set -> update tauk
                    # update tau
                    self.tauk = 7*self.tauk 

        elif (self.methodparam == 'penalty-lagrange'):

            # optimize: meritfunction(x) + lambda^T h(x) + 0.5*tau*||h(x)||_2^2

            print('----------------- run penalty lagrange -----------------')

            # choose the initial lamda
            self.lamk = numpy.ones(2*len(x0))

            xk_1 = x0
            xk   = x0
            while (1): 
             
                # define the gradient for the penalty-lagrange-function
                if (self.stochagradparam == True):
                    def stochagrad_total(x):
                        res = self.stochagrad(self.func, x, h) + \
                              grad_lag(x,self.bdry,self.tauk,self.lamk)
                        return res
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_lag(x,self.bdry,self.tauk,self.lamk)
                        return res
                    self.options['stochagrad'] = stochagrad_total
                else: # this is the default case
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_lag(x,self.bdry,self.tauk,self.lamk)
                        return res

                # store the grad-function in options
                self.options['grad'] = grad_total
            
                # update iteration number
                iterNum += 1

                print('\niteration number = %d' % (iterNum))
                printArray('bdry =', self.bdry)
                print('\ntau = %6.3f' % (self.tauk))
                printArray('lambda_k =', self.lamk)
                #printArray('gradient of the penalty term =',\
                #           grad_lag(xk,self.bdry, self.tauk, self.lamk))


                # find local minimizer of 
                # meritfunction(x) + (lambda_k)^T h(x) + 0.5*tau_k*||h(x)||_2^2
                penalty_lag_func = lambda x: self.func(x) +\
                                       0.5*self.tauk*numpy.square(\
                                       numpy.linalg.norm(eval_h(x,self.bdry)))+\
                                       numpy.dot(self.lamk,eval_h(x,self.bdry))

                res = minimize(penalty_lag_func,
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               jac=grad_total,
                               options=self.options, 
                               **self.kwargs)
                # update xk
                xk = res.x

                printArray('x_k =', res.x, point=points)
                print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))
                break

                # check if xk is in the feasible set with ||h(x)||_inf < 10*eps
                if (iterNum>=5):
                    print('\n----------- end of penalty lagrange run -----------')
                    print('\nReached max. numbers of penalty iterations = %d' % \
                          (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))<self.gtol) and
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('\n---------- end of penalty lagrange run ----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))>self.gtol) and 
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('Algorithm did not reach a Minimum but the solution is still in the feasible area!')
                    print('Keep Lambda and Tau and continue with Algorithm')
                else: # xk is not in the feasible set -> update tauk and lambdak
                    # update tau
                    self.tauk = 7*self.tauk
                    # update lambda
                    self.lamk = numpy.add(self.lamk, 
                                          self.tauk*eval_h(xk, self.bdry))
                # update xk-1
                xk_1 = xk


        elif (self.methodparam == 'log'):
            # Logarithmic Barrier Method
            print('----------------- run log barrier -----------------')

            self.my = 1.0 # '.0' is important, otherwise its an integer and my=0 
                          # in second step!

            xk_1 = x0
            xk   = x0
            while (1):
                # define the gradient for the penalty-lagrange-function
                if (self.stochagradparam == True):
                    def stochagrad_total(x):
                        res = self.stochagrad(self.func, x, h) + \
                              grad_log(x,self.bdry,self.my)
                        return res
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_log(x,self.bdry,self.my)
                        return res
                    self.options['stochagrad'] = stochagrad_total
                else: # this is the default case
                    def grad_total(x):
                        res = grad(self.func, x, h) + \
                              grad_log(x,self.bdry,self.my)
                        return res

                # store the grad-function in options
                self.options['grad']= grad_total

                # update iteration number
                iterNum += 1
                print('\niteration number = %d' % (iterNum))

                # find local minimizer of for the barrier method
                log_func = lambda x: self.func(x) - self.my*numpy.sum(\
                                     my_log(eval_c(x,self.bdry)))

                res = minimize(log_func,
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               options=self.options, 
                               **self.kwargs)
                # update xk
                xk = res.x
                
                # why do you need this line?
                normneu = numpy.linalg.norm(xk)

                printArray('x_k =', res.x, point=points)
                print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))

                # check if xk is in the feasible set with ||h(x)||_inf < 10*eps
                if (iterNum>=5):
                    print('\n----------- end of log run -----------')
                    print('\nReached max. numbers of penalty iterations = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))<self.gtol) and
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('\n---------- end of log run ----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                elif ((numpy.linalg.norm(grad_total(xk))>self.gtol) and 
                    (numpy.linalg.norm(eval_h(xk,self.bdry),numpy.inf)<tol_seq)):
                    print('Algorithm did not reach a Minimum but the solution is still in the feasible area!')
                    print('Keep my and continue with Algorithm')
                    break
                else: # xk is not in the feasible set -> update my
                    # update my
                    self.my = self.my/10

                # update xk-1
                xk_1 = xk

        else:
            raise ValueError('Methodparam is not found!')

        printArray('x_k =', res.x, point=points)
        print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))
        self.res = res
        return res.x


def sgd(func, x0, args, 
        maxiter=500,
        stepsize=1e-9,
        methods='vanilla',
        gamma=0.9,
        gradtol=1500,
        pathf=False,
        plot=False,
        plotset={},
        **kwargs):

    '''
    Stochastic gradient descent with options. 

    Input:
    func     - function to be optimized
    x0       - initial value
    args     - additional arguments for the func, is ignored within this algo, 
               but scipy requires this
    maxiter  - maximum number of iterations
    stepsize - stepsize for the update of x
    methods  - the method options, possible:
               vanilla : sgd
               momentum: sgd with momentum
               nag     : nesterovs accelerated gradient
               gradient: gradient decent with real gradient
    gamma    - parameter for the momentum
    gradtol  - tolerance for real gradient termination condition
    pathf    - return path of func if it is true
    plot     - plot iterations-functionvalues if it is true
    plotset  - plot settings

    Output: OptimizeResult-object from scipy

    Source: 
    Ruder, Sebastian. "An overview of gradient descent optimization algorithms."
    arXiv preprint arXiv:1609.04747 (2016).
    '''
    gradient   = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum    = 0
    xk_1       = x0
    fk_1       = func(x0)
    path       = numpy.empty(maxiter+1)
    path[0]    = fk_1

    # momentum vector
    vk_1 = numpy.zeros(len(x0))

    while (1):
        # update iteration number
        iternum += 1

        # choose the method
        if (methods == 'vanilla'):
            # vanilla sgd
            vk   = stepsize*stochagrad(xk_1)
        if (methods == 'momentum'):
            # momentum vector
            vk   = gamma*vk_1+stepsize*stochagrad(xk_1)
            vk_1 = vk
        if (methods == 'nag'):
            # nesterov accelerated gradient
            vk   = gamma*vk_1+stepsize*stochagrad(xk_1-gamma*vk_1)
            vk_1 = vk
        if (methods == 'gradient'):
            # gradient descent
            vk   = stepsize*gradient(xk_1)

        # iteration rule
        xk = xk_1 - vk 
        # debugging
        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)

        # debugging
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))
#        printArray('gk =', gk)

        # update path
        path[iternum] = fk
        # termination
        if ((iternum >= maxiter) or (gknorm <= gradtol)):
            break
        xk_1 = xk
        fk_1 = fk

    # cut off the path
    if not (iternum==maxiter):
        path = numpy.delete(path, range(iternum+1, maxiter)) 

    # return path of func if desired
    if (pathf==True):
        result = OptimizeResult(fun=path, x=xk, nit=iternum)
    else:
        result = OptimizeResult(fun=fk, x=xk, nit=iternum)

    # plot if desired
    if (plot==True):
        plot2d(range(len(path)), path, **plotset)


    return result


def adam(func, x0, args, 
         maxiter=300,
         stepsize=1e-2,
         beta1=0.1,
         beta2=0.99,
         epsilon=1e-2,
         gradtol=1500,
         pathf=False,
         plot=False,
         plotset={},
         **kwargs):

    '''
    Adaptive moment estimation - based on sgd

    Input:
    func     - function to be optimized
    x0       - initial value
    args     - additional arguments for the func, is ignored within this algo, 
               but scipy requires this
    maxiter  - maximum number of iterations
    stepsize - stepsize for the update of x
    beta1    - parameter for the momentum
    beta2    - parameter for the momentum
    epsilon  - parameter for the momentum
    gradtol  - tolerance for real gradient termination condition
    pathf    - return path of func if it is true
    plot     - plot iterations-functionvalues if it is true
    plotset  - plot settings

    Output: OptimizeResult-object from scipy

    Source: 
    Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic 
    optimization." arXiv preprint arXiv:1412.6980 (2014).
    '''

    gradient   = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum    = 0
    xk_1       = x0
    fk_1       = func(x0)
    path       = numpy.empty(maxiter+1)
    path[0]    = fk_1

    # 1st moment vector
    mk_1 = numpy.zeros(len(x0))
    # 2nd moment vector
    vk_1 = numpy.zeros(len(x0))

    while (1):
        # update iteration number
        iternum += 1
        # get stochastic gradient
        gk = stochagrad(xk_1)
        # update first moment estimate
        mk   = beta1*mk_1+(1-beta1)*gk
        mk_1 = mk
        # update second moment estimate
        vk   = beta2*vk_1+(1-beta2)*numpy.power(gk,2)
        vk_1 = vk
        # compute bias-corrected first moment estimate
        mk_hat = mk/(1-numpy.power(beta1,iternum))
        # compute bias-corrected first moment estimate
        vk_hat = vk/(1-numpy.power(beta2,iternum))

#        printArray('stepsize*mk_hat/(sqrt(vk_hat)+epsilon) =',stepsize*mk_hat/(numpy.power(vk_hat,0.5)+epsilon))

        # iteration rule
        xk = xk_1 - stepsize*mk_hat/(numpy.power(vk_hat,0.5)+epsilon)

        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))

        # update path
        path[iternum] = fk
        # termination
        if ((iternum >= maxiter) or (gknorm <= gradtol)):
            break
        xk_1 = xk
        fk_1 = fk

    # cut off the path
    if not (iternum==maxiter):
        path = numpy.delete(path, range(iternum+1, maxiter)) 

    # return path of func if desired
    if (pathf==True):
        result = OptimizeResult(fun=path, x=xk, nit=iternum)
    else:
        result = OptimizeResult(fun=fk, x=xk, nit=iternum)

    # plot if desired
    if (plot==True):
        plot2d(range(len(path)), path, **plotset)

    return result


def adamax(func, x0, args, 
           maxiter=300,
           stepsize=1e-2,
           beta1=0.09,
           beta2=0.99,
           gradtol=1500,
           pathf=False,
           plot=False,
           plotset={},
           **kwargs):

    '''
    Adaptive moment estimation - based on Adam: Adam generalized to p-norms 

    Input:
    func     - function to be optimized
    x0       - initial value
    args     - additional arguments for the func, is ignored within this algo, 
               but scipy requires this
    maxiter  - maximum number of iterations
    stepsize - stepsize for the update of x
    beta1    - parameter for the momentum
    beta2    - parameter for the momentum
    gradtol  - tolerance for real gradient termination condition
    pathf    - return path of func if it is true
    plot     - plot iterations-functionvalues if it is true
    plotset  - plot settings

    Output: OptimizeResult-object from scipy

    Source: 
    Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic 
    optimization." arXiv preprint arXiv:1412.6980 (2014).
    '''

    gradient   = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum    = 0
    xk_1       = x0
    fk_1       = func(x0)
    path       = numpy.empty(maxiter+1)
    path[0]    = fk_1

    # 1st moment vector
    mk_1 = numpy.zeros(len(x0))
    # 2nd moment vector
    vk_1 = numpy.zeros(len(x0))

    while (1):
        iternum += 1
        # get stochastic gradient
        gk = stochagrad(xk_1)
        # update first moment estimate
        mk   = beta1*mk_1+(1-beta1)*gk
        mk_1 = mk
        # update second moment estimate
        vk   = numpy.maximum(beta2*vk_1, numpy.absolute(gk))
        vk_1 = vk
        # iteration rule
        xk = xk_1 - (stepsize/(1-numpy.power(beta1,iternum)))*mk/vk

        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))
#        printArray('delta_x =', (stepsize/(1-numpy.power(beta1,iternum)))*mk/vk)


        # update path
        path[iternum] = fk
        # termination
        if ((iternum >= maxiter) or (gknorm <= gradtol)):
            break
        xk_1 = xk
        fk_1 = fk

    # cut off the path
    if not (iternum==maxiter):
        path = numpy.delete(path, range(iternum+1, maxiter)) 

    # return path of func if desired
    if (pathf==True):
        result = OptimizeResult(fun=path, x=xk, nit=iternum)
    else:
        result = OptimizeResult(fun=fk, x=xk, nit=iternum)

    # plot if desired
    if (plot==True):
        plot2d(range(len(path)), path, **plotset)


    return result


def adagrad(func, x0, args, 
            maxiter=300,
            stepsize=1e-3,
            epsilon=1e-3,
            gradtol=1500,
            pathf=False,
            plot=False,
            plotset={},
            **kwargs):

    '''
    Adaptive gradient estimation

    Input:
    func     - function to be optimized
    x0       - initial value
    args     - additional arguments for the func, is ignored within this algo, 
               but scipy requires this
    maxiter  - maximum number of iterations
    stepsize - stepsize for the update of x
    epsilon  - parameter for the gradient
    gradtol  - tolerance for real gradient termination condition
    pathf    - return path of func if it is true
    plot     - plot iterations-functionvalues if it is true
    plotset  - plot settings

    Output: OptimizeResult-object from scipy

    Source: 
    Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for
    online learning and stochastic optimization." Journal of Machine Learning 
    Research 12.Jul (2011): 2121-2159.
    '''

    gradient   = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum    = 0
    dim        = len(x0)
    xk_1       = x0
    fk_1       = func(x0)
    G          = numpy.zeros(dim)
    path       = numpy.empty(maxiter+1)
    path[0]    = fk_1

    while (1):
        # update iteration number
        iternum += 1
        # get stochastic gradient
        gk = stochagrad(xk_1)
        # update G
        G += gk*gk
        # iteration rule
        xk = xk_1 - stepsize*(gk/(numpy.sqrt(G)+epsilon))

        # debugging
        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))
#        printArray('gk =', gk)

        # update path
        path[iternum] = fk
        # termination
        if ((iternum >= maxiter) or (gknorm <= gradtol)):
            break
        xk_1 = xk
        fk_1 = fk

    # cut off the path
    if not (iternum==maxiter):
        path = numpy.delete(path, range(iternum+1, maxiter)) 

    # return path of func if desired
    if (pathf==True):
        result = OptimizeResult(fun=path, x=xk, nit=iternum)
    else:
        result = OptimizeResult(fun=fk, x=xk, nit=iternum)

    # plot if desired
    if (plot==True):
        plot2d(range(len(path)), path, **plotset)


    return result


def adadelta(func, x0, args, 
             maxiter=300,
             roh=0.999,
             epsilon=1e-8,
             gradtol=1500,
             pathf=False,
             plot=False,
             plotset={},
             **kwargs):

    '''
    Adaptive learning rate methode (learning rate = stepsize)

    Input:
    func     - function to be optimized
    x0       - initial value
    args     - additional arguments for the func, is ignored within this algo, 
               but scipy requires this
    maxiter  - maximum number of iterations
    epsilon  - parameter for the gradient
    gradtol  - tolerance for real gradient termination condition
    pathf    - return path of func if it is true
    plot     - plot iterations-functionvalues if it is true
    plotset  - plot settings

    Output: OptimizeResult-object from scipy

    Source: 
    Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv 
    preprint arXiv:1212.5701 (2012).
    '''

    gradient   = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum    = 0
    dim        = len(x0)
    xk_1       = x0
    fk_1       = func(x0)
    Egk_1      = numpy.zeros(dim)
    Exk_1      = numpy.zeros(dim)
    path       = numpy.empty(maxiter+1)
    path[0]    = fk_1

    while (1):
        # update iteration number
        iternum += 1
        # get stochastic gradient
        gk = stochagrad(xk_1)
        # accumulate gradient
        Egk   = roh*Egk_1+(1-roh)*gk*gk
        Egk_1 = Egk
        # calculate RMS[g]_k and RMS[delta x]_k-1
        RMSg = numpy.sqrt(Egk  +epsilon)
        RMSx = numpy.sqrt(Exk_1+epsilon)
        # delta
        delta_xk = -(RMSx/RMSg)*gk
        # iteration rule
        xk = xk_1 + delta_xk
        # accumulate updates
        Exk   = roh*Exk_1+(1-roh)*delta_xk*delta_xk
        Exk_1 = Exk

        # debugging
        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))

        # update path
        path[iternum] = fk
        # termination
        if ((iternum >= maxiter) or (gknorm <= gradtol)):
            break
        xk_1 = xk
        fk_1 = fk

    # cut off the path
    if not (iternum==maxiter):
        path = numpy.delete(path, range(iternum+1, maxiter)) 

    # return path of func if desired
    if (pathf==True):
        result = OptimizeResult(fun=path, x=xk, nit=iternum)
    else:
        result = OptimizeResult(fun=fk, x=xk, nit=iternum)

    # plot if desired
    if (plot==True):
        plot2d(range(len(path)), path, **plotset)

    return result


def get_scipy_stochastic_hybrid(stocha_opt_func, scipy_opt_func):

    '''
    Method to generate a hybrid optimization algorithm in a generic fashion.

    Input:
    stocha_opt_func : A desired stochastic optimization function, which is then
                      used to get in a surrounding of a minimum (pointer to a 
                      function)
    scipy_opt_func  : A desired scipy optimization method, which is then used to
                      generate convergence to the minimum (string of a scipy 
                      method)

    Output: hybrid optimization algorithm 
    '''

    def scipy_stochastic_hybrid(func, x0, args=(), 
                                options_s={},
                                options_d={},
                                plot=False,
                                **kwargs):

        '''
        A hybrid optimization algorithm
 
        Input:
        func      - function to be optimized
        x0        - initial value
        args      - additional arguments for the func, is ignored within this 
                    algo, but scipy requires this
        options_s - solver options for the stochastic optimization function
        options_d - solver options for the deterministic optimization function
        plot      - iterations-functionvalue-plot if it is true

        Output: OptimizeResult-object from scipy
        '''

        if (len(options_s)==0):
            raise ValueError('There are no solver options for the stochastic \
                              method')
        
        if (len(options_d)==0):
            raise ValueError('There are no solver options for the scipy \
                              method')

        if (kwargs.get('stochagrad', None)==None):
            raise ValueError('Stochastic gradient is required!')

        print('---------------- start stochastic method ----------------')
        # stochastic method to get in a surrounding of a minimum
        res_approx = stocha_opt_func(func, x0, args, 
                                     grad      =kwargs['grad'],
                                     stochagrad=kwargs['stochagrad'],
                                     pathf=plot,
                                     **options_s)

        print('-------------- start deterministic method ---------------')
        # deterministic method to get convergence
        res_sol = minimize(fun=func, x0=res_approx.x, args=args,
                           method =scipy_opt_func,
                           jac    =kwargs['jac'],
                           options=options_d) 

        # set the result
        iternum = res_approx.nit + res_sol.nit
        fun     = numpy.append(res_approx.fun, res_sol.fun)
        result  = OptimizeResult(fun=fun, x=res_sol.x, nit=iternum)

        # plot if desired
        if (plot==True):
            plot2d(range(len(fun)), fun, ylog=True, save=True)
        
        return result
    
    return scipy_stochastic_hybrid


def gradient_descent(func, x0, args=(),
                     maxiter=100, stepsize=1e-8, **kwargs):
    xk = x0
    grad = kwargs['grad']
    iternum = 0 

    while (iternum < maxiter):
        iternum += 1
        xk -= stepsize*grad(xk)
        if (numpy.linalg.norm(grad(xk), numpy.inf) <= 500):
            print('\ngradient is near zero\n')
            print('||grad(x_final)||_inf = %10.6'\
                  % (numpy.linalg.norm(grad(xk), numpy.inf)))
            print('\nTerminated in iteration: %d' % (iternum))
            break

    printArray('gradient in gradient decsent for x_final = ', grad(xk))

    return OptimizeResult(fun=func(xk), x=xk, nit=iternum)


def plot2d_meritfunction(func, x0, args=(), 
                         disk=100,
                         plotvar=0,
                         interval=None,
                         plotset={},
                         **kwargs):

    mf = numpy.empty(disk)
    dim = len(x0)
    x = x0
    if (plotvar >= dim or 0 > plotvar):
        raise ValueError('plotvar is out of range [0,len(x0)-1]')

    lb = kwargs['bounds'].lb[plotvar]
    ub = kwargs['bounds'].ub[plotvar]
    if type(interval) is not numpy.ndarray:
        if (interval == None):
            interval = numpy.linspace(lb, ub, disk)
        else:
            raise ValueError('Something is wrong with interval')
    else:
        interval = numpy.linspace(interval[0], interval[1], disk)

    for i in range(disk):
        x[plotvar] = interval[i]
        mf[i] = func(x)
    
    plot2d(interval, mf, **plotset)

    return OptimizeResult(fun=func(x), x=x, nit=disk)


def plot3d_meritfunction(func, x0, args=(), 
                         disk=100,
                         plotvar1=0,
                         plotvar2=1,
                         interval1=None,
                         interval2=None,
                         plotset={},
                         **kwargs):

    mf = numpy.empty((disk, disk))
    dim = len(x0)
    x = x0

    if (plotvar1 >= dim or 0 > plotvar1):
        raise ValueError('plotvar[0] is out of range [0,len(x0)-1]')

    if (plotvar2 >= dim or 0 > plotvar2):
        raise ValueError('plotvar[1] is out of range [0,len(x0)-1]')

    lb1 = kwargs['bounds'].lb[plotvar1]
    ub1 = kwargs['bounds'].ub[plotvar1]
    lb2 = kwargs['bounds'].lb[plotvar2]
    ub2 = kwargs['bounds'].ub[plotvar2]

    if type(interval1) is not numpy.ndarray:
        if (interval1 == None):
            interval1 = numpy.linspace(lb1, ub1, disk)
        else:
            raise ValueError('Something is wrong with interval1')
    else:
        interval1 = numpy.linspace(interval1[0], interval1[1], disk)

    if type(interval2) is not numpy.ndarray:
        if (interval2 == None):
            interval2 = numpy.linspace(lb2, ub2, disk)
        else:
            raise ValueError('Something is wrong with interval2')
    else:
        interval2 = numpy.linspace(interval2[0], interval2[1], disk)

    for i in range(disk):
        x[plotvar1] = interval1[i]
        for j in range(disk):
            x[plotvar2] = interval2[j]
            mf[j,i] = func(x)
    
    X, Y = numpy.meshgrid(interval1, interval2)
    plot3d(X, Y, mf, **plotset)

    return OptimizeResult(fun=func(x), x=x, nit=disk)



def test_minimize_neldermead(func, x0, args=(), 
                             maxiter=100, 
                             xatol=1e-6, fatol=1e-6,
                             **unknown_options):
    alpha = 1
    beta = 2
    gamma = 0.5
    sigma = 0.5
    # convert x into float array and flat
    x0 = numpy.asfarray(x0).flatten()
    N = len(x0)
    nonzdelt = 0.05 # params from scipy impl
    zdelt = 0.00025 # params from scipy impl
    # set up simplex, also from scipy impl in order to get the same init simplex
    sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
    sim[0] = x0
    for k in range(N):
        y = numpy.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y

    # initilize f simplex
    fsim = numpy.zeros((N + 1,), float)
    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = numpy.argsort(fsim)
    # sort f simplex so fsim [0,:] is the lowest function value
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)
    iterNum = 1
    # we follow the implementation from the num3 lecture notes
    while (iterNum < maxiter):
        # same break condition as scipy
        if (numpy.max(numpy.ravel(numpy.abs(sim[1:] - sim[0]))) <= xatol and
                numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= fatol):
            break
        xbar = numpy.add.reduce(sim[:-1],0)/N
        x_a = xbar - alpha*(sim[-1]-xbar)
        f_a = func(x_a)

        if (fsim[0] <= f_a and f_a <= fsim[-2]):
            sim[-1] = x_a
        elif (f_a < fsim[0]):
            x_ab = xbar - alpha*beta*(sim[-1]-xbar)
            f_ab = func(x_ab)
            if (f_ab < fsim[0]):
                sim[-1] = x_ab
            else: 
                sim[-1] = x_a
        else: # f_a > fsim[-2]
            if (fsim[-2] < f_a and f_a < fsim[-1]):
                x_ag = xbar - alpha*gamma*(sim[-1]-xbar)
                f_ag = func(x_ag)
                if (f_ag <= f_a):
                    sim[-1] = x_ag
                else: 
                    for i in range(N):
                        sim[i+1] = sigma*(sim[0]+sim[i+1])
            else: # f_a is the worst value
                xg = xbar + gamma*(sim[-1]-xbar)
                fg = func(xg)
                if (fg < fsim[-1]):
                    sim[-1] = xg
                else:
                    for i in range(N):
                        sim[i+1] = sigma*(sim[0]+sim[i+1])
        for i in range(N+1):
            fsim[i] = func(sim[i])

        iterNum += 1
        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)

    x_sol = sim[0]
    fval = numpy.min(fsim)
    return OptimizeResult(fun=fval, x=x_sol, nit=iterNum, 
                          final_simplex=(sim, fsim))


def Nelder_Mead_Constraint(func,x0,args=(),maxiter=100,tol=1e-5,\
                            **unknown_options):
    #---------------------------------------------------------------------------
    def Cf(x,lb,ub):                        # Constraint fitness priority-based ranking method
        C = numpy.empty(len(lb)+len(ub))
        g = numpy.empty(len(lb)+len(ub))
        for t in range(len(x)):
            g[2*t] = lb[t]-x[t]             # value of g_i(x)
            g[2*t+1] = x[t]-ub[t]
        gmax = numpy.amax(g)
        for t in range(len(g)):
            if (g[t]<=0):
                C[t] = 1
            else:
                C[t] = 1-(g[t]/gmax)
        weight = 1/float((len(C)))
        Cf = 0
        for t in range(len(C)):
            Cf += weight*C[t]
        return Cf
    #---------------------------------------------------------------------------
    # Parameter:
    alpha = 1
    beta = 0.5
    gamma = 2
    delta = 0.5
    tol = 1e-12         # for check if sth x==0 -> check x<tol and x>-tol
    
    bounds = unknown_options['bounds']
    lb = bounds.lb
    ub = bounds.ub
    n = len(x0)
    # generate initial_simplex:
    initial_simplex = numpy.empty([n+1,n])
    for t in range(n+1):
        initial_simplex[t] = numpy.random.uniform(bounds.lb,bounds.ub) #in feasible region
    xfC = []                    # list with [[x1,f(x1),Cf(x1)],[x2,f(x2),Cf(x2)],...
    for t in range(n+1):
        xfC.append([initial_simplex[t],func(initial_simplex[t]),\
                    Cf(initial_simplex[t],bounds.lb,bounds.ub)])
    
    # termination condition for first check in while loop:
    f_mean = 0
    for t in range(n+1):
        f_mean += xfC[t][1]
    f_mean = f_mean * (1/(float(n)+1))
    error = 0
    for t in range(n+1):
        error += (xfC[t][1]-f_mean)**2
    error *= 1/(float(n)+1)

    it = 0
    # sort xfC from low to high f(x) values
    xfC = sorted(xfC,key=lambda elem: elem[1])
    while (it <= maxiter and error > tol):
        # Reflection:
        # Determine x_cent without x_high:
        x_cent = numpy.zeros(n)
        for t in range(n):
            for j in range(n):
                x_cent[j] += xfC[t][0][j]
        for t in range(n):
            x_cent[t] /= n
        # make reflection:
        x_high = xfC[n][0]
        f_high = xfC[n][1]
        x_refl = (1+alpha)*x_cent-alpha*x_high
        Cf_refl = Cf(x_refl,lb,ub)
        f_refl = func(x_refl)
        x_low = xfC[0][0]
        Cf_low = xfC[0][2]
        f_low = xfC[0][1]
        

        if ((Cf_refl<1 and Cf_refl>Cf_low) or ((Cf_refl<=1+tol and Cf_refl>=1-tol) and \
            f_refl<f_low)):
            # first expansion case:
            if (Cf_refl<1 and Cf_refl>Cf_low):
                x_exp = gamma*x_refl+(1-gamma)*x_cent
                f_exp = func(x_exp)
                Cf_exp = Cf(x_exp,lb,ub)
                if (Cf_exp>Cf_low or (Cf_exp>=1-tol and Cf_exp<=1+tol)):
                    # expansion accepted -> replace x_high by x_exp:
                    xfC[n][0] = x_exp
                    xfC[n][1] = f_exp
                    xfC[n][2] = Cf_exp
                else:
                    # replace x_high by x_refl:
                    xfC[n][0] = x_refl
                    xfC[n][1] = f_refl
                    xfC[n][2] = Cf_refl
            # second expansion case:
            if ((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl<f_low)):
                x_exp = gamma*x_refl+(1-gamma)*x_cent
                f_exp = func(x_exp)
                Cf_exp = Cf(x_exp,lb,ub)
                if ((Cf_exp>=1-tol and Cf_exp<=1+tol)and(f_exp<f_low)):
                    # expansion ist acceptet, replace x_high by x_exp:
                    xfC[n][0] = x_exp
                    xfC[n][1] = f_exp
                    xfC[n][2] = Cf_exp
                else:
                    # replacing x_high by x_refl:
                    xfC[n][0] = x_refl
                    xfC[n][1] = f_refl
                    xfC[n][2] = Cf_refl

        else:
            # first contraction case:
            if (Cf_refl<1 and Cf_refl<=Cf_low):
                x_cont = beta*x_high+(1-beta)*x_cent
                f_cont = func(x_cont)
                Cf_cont = Cf(x_cont,lb,ub)
                if ((Cf_cont>Cf_low)or(Cf_cont>=1-tol and Cf_cont<=1+tol)):
                    # contraction is accepted: replace x_high by x_cont:
                    xfC[n][0] = x_cont
                    xfC[n][1] = f_cont
                    xfC[n][2] = Cf_cont
                else:
                    # shrinkage attempts to all points except x_low:
                    for t in range(n):
                        xfC[1+t][0] *= delta
                        xfC[1+t][0] += (1-delta)*x_low
            # second contraction case:
            if((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl>=f_low) \
                and (f_refl<=f_high)):
                # replace x_high by x_refl:
                xfC[n][0] = x_refl
                xfC[n][1] = f_refl
                xfC[n][2] = Cf_refl
                x_cont = beta*x_refl+(1-beta)*x_cent
                f_cont = func(x_cont)
                Cf_cont = Cf(x_cont,lb,ub)
                if((Cf_cont>=1-tol and Cf_cont<=1+tol)and(f_cont<f_low)):
                    # contraction accepted: replace x_high by x_cont
                    xfC[n][0] = x_cont
                    xfC[n][1] = f_cont
                    xfC[n][2] = Cf_cont
                else:
                    # shrinkage the entire simplex but not x_low
                    for t in range(n):
                        xfC[1+t][0] *= delta
                        xfC[1+t][0] += (1-delta)*x_low
            # third contraction case:
            if((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl>=f_low) \
                and (f_refl>f_high)):
                x_cont = beta*x_high+(1-beta)*x_cent
                f_cont = func(x_cont)
                Cf_cont = Cf(x_cont,lb,ub)
                if((Cf_cont>=1-tol and Cf_cont<=1+tol)and(f_cont<f_low)):
                    # contraction accepted: replace x_high by x_cont
                    xfC[n][0] = x_cont
                    xfC[n][1] = f_cont
                    xfC[n][2] = Cf_cont
                else:
                    # shrinkage the entire simplex but not x_low
                    for t in range(n):
                        xfC[1+t][0] *= delta
                        xfC[1+t][0] += (1-delta)*x_low
        # sort xfC from low to high f(x) values
        xfC = sorted(xfC,key=lambda elem: elem[1])
        it = it+1
        # termination condition:
        f_mean = 0
        for t in range(n+1):
            f_mean += xfC[t][1]
        f_mean *= 1/(float(n)+1)
        error = 0
        for t in range(n+1):
            error += (xfC[t][1]-f_mean)**2
        error *= 1/(float(n)+1)   

    final_simplex = numpy.empty([n+1,n])
    for t in range(n+1):
        final_simplex[t] = xfC[t][0]
    return OptimizeResult(fun=xfC[0][1], x=final_simplex[0], nit=it,\
                            final_simplex=final_simplex)


def PSO_NM_1(func,x0,args=(),maxiter=100,c0=None,c1=2.0,c2=2.0,\
             typ=1,S=25,\
             maxIt_NM=100,tol_NM=1e-6,\
             improve=True,\
             stopNumber=10, stopTol=0.1,**unknown_options):
    # Typ 1: Updating with global best and best of 2-neighborhood
    # Typ 2: Updating with global best and individual best since start
    # Typ 3: Population is divided in S Subpopulations
    # Typ 4: same as typ 1 but in PSO all particles are updated
    # c0,c1 and c2 are variables for updating the velocity of the particles
    # S: number of neighborhoods/subpopulations
    # MaxIt_NM: max number of iterations for Nelder-Mead
    # tol_NM: stopping criteria for Nelder-Mead (tolerance)
    # improve: if true, "TNC" is applied to best find solution to improve solution
    # if fBest doesn't decrease about (stopTol*100)% in stopNumber iterations -> stop
    
    # subfunctions needed:
    #---------------------------------------------------------------------------
    def rm(x,lb,ub,vel_max):                      # repair method (later grad.bas.r.m)
        # moves particles on bounds if they don't be in the feasible region
        from copy import deepcopy
        x_neu = deepcopy(x)
        for t in range(len(x)):
            if (x[t]<lb[t]):
                x_neu[t] = lb[t] 
            if (x[t]>ub[t]):
                x_neu[t] = ub[t] 
        return x_neu

    def Cf(x,lb,ub):                        # Constraint fitness priority-based ranking method
        C = numpy.empty(len(lb)+len(ub))
        g = numpy.empty(len(lb)+len(ub))
        for t in range(len(x)):
            g[2*t] = lb[t]-x[t]             # value of g_i(x)
            g[2*t+1] = x[t]-ub[t]
        gmax = numpy.amax(g)
        for t in range(len(g)):
            if (g[t]<=0):
                C[t] = 1
            else:
                C[t] = 1-(g[t]/gmax)
        weight = 1/float((len(C)))
        Cf = 0
        for t in range(len(C)):
            Cf += weight*C[t]
        return Cf

    def nelder_mead_con(initial_simplex,func,Cf,lb,ub):
    # Nelder-Mead for constraint optimization
    # initial_simplex should be array of shape (N+1,N) with N: problem size
        alpha = 1
        beta = 0.5
        gamma = 2
        delta = 0.5
        tol = 1e-10        

        n = len(initial_simplex[0])
        xfC = []                    # list with [[x1,f(x1),Cf(x1)],[x2,f(x2),Cf(x2)],...
        for t in range(n+1):
            xfC.append([initial_simplex[t],func(initial_simplex[t]),\
                        Cf(initial_simplex[t],lb,ub)])
        
        # termination condition for first check in while loop:
        f_mean = 0
        for t in range(n+1):
            f_mean += xfC[t][1]
        f_mean = f_mean * (1/(float(n)+1))
        error = 0
        for t in range(n+1):
            error += (xfC[t][1]-f_mean)**2
        error *= 1/(float(n)+1)

        it = 1

        # sort xfC from low to high f(x) values
        xfC = sorted(xfC,key=lambda elem: elem[1])

        while (it <= maxIt_NM and error > tol_NM):
            ## sort xfC from low to high f(x) values
            #xfC = sorted(xfC,key=lambda elem: elem[1]) # kann geloescht werden
            # Reflection:
            # Determine x_cent without x_high:
            x_cent = numpy.zeros(n)
            for t in range(n):
                for j in range(n):
                    x_cent[j] += xfC[t][0][j]
            for t in range(n):
                x_cent[t] /= n
            # make reflection:
            x_high = xfC[n][0]
            f_high = xfC[n][1]
            x_refl = (1+alpha)*x_cent-alpha*x_high
            Cf_refl = Cf(x_refl,lb,ub)
            f_refl = func(x_refl)
            x_low = xfC[0][0]
            Cf_low = xfC[0][2]
            f_low = xfC[0][1]
            
    
            if ((Cf_refl<1 and Cf_refl>Cf_low) or ((Cf_refl<=1+tol and Cf_refl>=1-tol) and \
                f_refl<f_low)):
                # first expansion case:
                if (Cf_refl<1 and Cf_refl>Cf_low):
                    x_exp = gamma*x_refl+(1-gamma)*x_cent
                    f_exp = func(x_exp)
                    Cf_exp = Cf(x_exp,lb,ub)
                    if (Cf_exp>Cf_low or (Cf_exp>=1-tol and Cf_exp<=1+tol)):
                        # expansion accepted -> replace x_high by x_exp:
                        xfC[n][0] = x_exp
                        xfC[n][1] = f_exp
                        xfC[n][2] = Cf_exp
                    else:
                        # replace x_high by x_refl:
                        xfC[n][0] = x_refl
                        xfC[n][1] = f_refl
                        xfC[n][2] = Cf_refl
                # second expansion case:
                if ((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl<f_low)):
                    x_exp = gamma*x_refl+(1-gamma)*x_cent
                    f_exp = func(x_exp)
                    Cf_exp = Cf(x_exp,lb,ub)
                    if ((Cf_exp>=1-tol and Cf_exp<=1+tol)and(f_exp<f_low)):
                        # expansion ist acceptet, replace x_high by x_exp:
                        xfC[n][0] = x_exp
                        xfC[n][1] = f_exp
                        xfC[n][2] = Cf_exp
                    else:
                        # replacing x_high by x_refl:
                        xfC[n][0] = x_refl
                        xfC[n][1] = f_refl
                        xfC[n][2] = Cf_refl
    
            else:
                # first contraction case:
                if (Cf_refl<1 and Cf_refl<=Cf_low):
                    x_cont = beta*x_high+(1-beta)*x_cent
                    f_cont = func(x_cont)
                    Cf_cont = Cf(x_cont,lb,ub)
                    if ((Cf_cont>Cf_low)or(Cf_cont>=1-tol and Cf_cont<=1+tol)):
                        # contraction is accepted: replace x_high by x_cont:
                        xfC[n][0] = x_cont
                        xfC[n][1] = f_cont
                        xfC[n][2] = Cf_cont
                    else:
                        # shrinkage attempts to all points except x_low:
                        for t in range(n):
                            xfC[1+t][0] *= delta
                            xfC[1+t][0] += (1-delta)*x_low
                # second contraction case:
                if((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl>=f_low) \
                    and (f_refl<=f_high)):
                    # replace x_high by x_refl:
                    xfC[n][0] = x_refl
                    xfC[n][1] = f_refl
                    xfC[n][2] = Cf_refl
                    x_cont = beta*x_refl+(1-beta)*x_cent
                    f_cont = func(x_cont)
                    Cf_cont = Cf(x_cont,lb,ub)
                    if((Cf_cont>=1-tol and Cf_cont<=1+tol)and(f_cont<f_low)):
                        # contraction accepted: replace x_high by x_cont
                        xfC[n][0] = x_cont
                        xfC[n][1] = f_cont
                        xfC[n][2] = Cf_cont
                    else:
                        # shrinkage the entire simplex but not x_low
                        for t in range(n):
                            xfC[1+t][0] *= delta
                            xfC[1+t][0] += (1-delta)*x_low
                # third contraction case:
                if((Cf_refl>=1-tol and Cf_refl<=1+tol)and(f_refl>=f_low) \
                    and (f_refl>f_high)):
                    x_cont = beta*x_high+(1-beta)*x_cent
                    f_cont = func(x_cont)
                    Cf_cont = Cf(x_cont,lb,ub)
                    if((Cf_cont>=1-tol and Cf_cont<=1+tol)and(f_cont<f_low)):
                        # contraction accepted: replace x_high by x_cont
                        xfC[n][0] = x_cont
                        xfC[n][1] = f_cont
                        xfC[n][2] = Cf_cont
                    else:
                        # shrinkage the entire simplex but not x_low
                        for t in range(n):
                            xfC[1+t][0] *= delta
                            xfC[1+t][0] += (1-delta)*x_low
            it = it+1
            # termination condition:
            f_mean = 0
            for t in range(n+1):
                f_mean += xfC[t][1]
            f_mean *= 1/(float(n)+1)
            error = 0
            for t in range(n+1):
                error += (xfC[t][1]-f_mean)**2
            error *= 1/(float(n)+1)   

            # sort xfC from low to high f(x) values
            xfC = sorted(xfC,key=lambda elem: elem[1])

        final_simplex = numpy.empty([n+1,n])
        for t in range(n+1):
            final_simplex[t] = xfC[t][0]
        return final_simplex
    
    #---------------------------------------------------------------------------

    n = len(x0)             # problem size
    bounds = unknown_options['bounds']

    N = 21*n+1              # swarm size (number of particles)
    
    vel_max = numpy.empty(n)    # define maximal verlocity
    for q in range(n):
        vel_max[q] = 0.1 * (bounds.ub[q]-bounds.lb[q])
    
    if (c0==None):          # define weight c0
        c0 = 0.5 + (random.random()/3)

    # Generate a class for the particles:
    class particle:
        def __init__(self,pos,vel):
            self.pos = pos
            self.pos_f = func(pos)
            self.vel = vel
            self.best = self.pos
            self.best_f = self.pos_f
            self.NH = random.randint(0,S-1)     # specifies to which neighborhood particle participate
        def updatePos(self,pos):
            self.pos = pos
            self.pos_f = func(self.pos)
            if(numpy.all(self.pos>=bounds.lb) and numpy.all(self.pos<=bounds.ub)):
                if (self.pos_f < self.best_f):
                    self.best_f = self.pos_f
                    self.best = self.pos
        def update(self,c0,c1,c2,gBest,nBest=numpy.empty(n)):  # gBest:global best; nBest:neighborhood best
            r1 = random.random()       # random number between 0 and 1
            r2 = random.random()
            if (typ == 1):
                self.vel = c0*self.vel + c1*r1*(nBest-self.pos) + \
                            c2*r2*(gBest-self.pos)
            elif (typ == 2):
                self.vel = c0*self.vel + c1*r1*(self.best-self.pos) + \
                            c2*r2*(gBest-self.pos)
            elif (typ == 3):
                self.vel = c0*self.vel + c1*r1*(self.best-self.pos) + \
                            c2*r2*(gBest-self.pos)  #gBest is neighborhood best
            if (typ == 4):
                self.vel = c0*self.vel + c1*r1*(nBest-self.pos) + \
                            c2*r2*(gBest-self.pos)           
            #check if self.vel is out of [-vel_max,vel_max]:
            for q in range(n):
                if (self.vel[q] < -vel_max[q]):
                    self.vel[q] = -vel_max[q]
                if (self.vel[q] > vel_max[q]):
                    self.vel[q] = vel_max[q]
            # Update position
            self.pos = self.pos + self.vel
            self.pos_f = func(self.pos)
            if(numpy.all(self.pos>=bounds.lb) and numpy.all(self.pos<=bounds.ub)):
                if (self.pos_f < self.best_f):
                    self.best_f = self.pos_f
                    self.best = self.pos   
        # reset particle (not NH neighborhood):
        def reset(self):
            self.pos = numpy.random.uniform(bounds.lb,bounds.ub)
            self.pos_f = func(self.pos)
            self.vel = numpy.random.uniform(-vel_max,vel_max)
            self.best = self.pos
            self.best_f = self.pos_f
    
    # Initialization of all particles with Position and velocity:
    swarm = []

    for i in range(N):
        pos0 = numpy.random.uniform(bounds.lb,bounds.ub)    # start position in feasible region
        vel0 = numpy.random.uniform(-vel_max,vel_max)       # random start velocity
        swarm.append(particle(pos0,vel0))

    k = 0                                                   # Iteration
    xBest = numpy.empty(n)
    xBest_f = 1e100

    xBest_old = numpy.zeros(n)      
    stopCtr = 0
    stopList = []

    while (k <= maxiter):
        #Evaluate solutions and apply Repair Method if not in feasible region:
        liste = []                            # list to safe (pos(i),f(pos(i)),...
        infeasible = 0
        for i in range(N):
            # check if solution is in feasibale region, else apply Repair Method
            if not (numpy.all(swarm[i].pos>=bounds.lb) and \
                    numpy.all(swarm[i].pos<=bounds.ub)):
                repairedPos = rm(swarm[i].pos,bounds.lb,bounds.ub,vel_max)
                swarm[i].updatePos(repairedPos)
                swarm[i].vel = numpy.zeros(n)       # vel=0 so that the particles doesn't go in infeasible region again
                infeasible += 1
            # write tupel (i,f(i)) in liste:
            temp = [i,swarm[i].pos_f]
            liste.append(temp)
        #print("repaired particles = ", infeasible)
        
        # sort liste from low func values to high func values:
        liste = sorted(liste,key=lambda elem: elem[1])
        # Update xBest and xBest_f if solution is better:
        if (liste[0][1] <= xBest_f):
            if (abs(liste[0][1]-xBest_f)>1e-3):  # check if global fBest is changing sifnificantly
                stopCrit = 0
            xBest = swarm[liste[0][0]].pos
            xBest_f = liste[0][1]
        
        print("bestes f = ", xBest_f)
        #print("Abstand zwischen alter und neuer Loesung: ", numpy.linalg.norm(xBest-xBest_old))
        xBest_old = xBest

        if len(stopList) < stopNumber:
            stopList.append(xBest_f)
            STOP = False
        if len(stopList) >= stopNumber:
            if (abs(stopList[0]-stopList[-1]) < stopTol*stopList[0]):
                STOP = True           
            stopList = stopList[1:len(stopList)]
        
        # check stopping criteria: ---------------------------------------------
        if (k==maxiter or STOP):
            print("Merit Final nach PSO_NM = ", xBest_f)
            print("xBest nach PSO_NM = ", xBest)
            print("Norm Gradient nach PSO_NM = ", \
                numpy.linalg.norm(grad(func,xBest,math.sqrt(numpy.finfo(float).eps))))
            
            if improve:
                def jaco(x):                                #needed for scipy algos
                    h = math.sqrt(numpy.finfo(float).eps)
                    return grad(func,x,h)
    
                Res = minimize(func,xBest,args=(),method='TNC',jac=jaco, \
                        bounds=bounds,options={"maxiter":500,"disp":True})
                print("Merit nach lokaler Suche = ", Res.fun)
                print("x nach lokaler Suche = ", Res.x)
                print("Norm Gradient nach lokaler Suche = ", \
                    numpy.linalg.norm(grad(func,Res.x,math.sqrt(numpy.finfo(float).eps))))
    
                if (Res.fun < xBest_f):
                    xBest_f = Res.fun
                    xBest = Res.x

            return OptimizeResult(fun=xBest_f, x=xBest, nit = k)
            break
        #-----------------------------------------------------------------------
        # Apply Nelder-Mead to the (n+1) best particles:
        initial_simplex = numpy.empty([n+1,n])
        for i in range(n+1):
            initial_simplex[i] = swarm[liste[i][0]].pos
        final_simplex = nelder_mead_con(initial_simplex,func,Cf,bounds.lb,bounds.ub)
        # Update (n+1)th particle:
        parti = liste[n][0]
        swarm[parti].updatePos(final_simplex[0])
        liste[n][1] = swarm[parti].pos_f

        # Update liste:
        liste = sorted(liste,key=lambda elem: elem[1])
        
        # Apply PSO:
        #1.) Determine global best particle:
        gBest = liste[0][0]             # index of global best particle
        
        if (typ == 1):
            # Divide 20n paritcles in two neighborhoods with 10N particles and
            # determine best of these twos:
            firstN = numpy.arange(n+1,10*n+n+1,dtype=int)
            secondN = numpy.arange(10*n+n+1,20*n+n+1,dtype=int)
            random.shuffle(firstN)          # random shuffeling
            random.shuffle(secondN)         # "
            for i in range(len(firstN)):
                first = firstN[i]
                second = secondN[i]
                nBest = 0
                if (liste[first][1] <= liste[second][1]):
                    nBest = liste[first][0]
                else:
                    nBest = liste[second][0]
                swarm[liste[first][0]].update(c0,c1,c2,gBest=swarm[gBest].pos,nBest=swarm[nBest].pos)     # update velocity and position
                swarm[liste[second][0]].update(c0,c1,c2,gBest=swarm[gBest].pos,nBest=swarm[nBest].pos)
        elif (typ == 2):
            for i in range(20*n):
                swarm[liste[n+1+i][0]].update(c0,c1,c2,gBest=swarm[gBest].pos)
        elif (typ == 3):
            for i in range(S):
                neighbor = []
                tempBest_f = 1e100
                for t in range(len(liste)):
                    if (swarm[liste[t][0]].NH==i):
                        neighbor.append(t)
                        if (swarm[liste[t][0]].best_f < tempBest_f):
                            tempBest_f = swarm[liste[t][0]].best_f
                            tempBest = liste[t][0]
                for t in range(len(neighbor)):
                    if (neighbor[t] >= n+1):      # only update particles which were not involved in Nelder-Mead-Update
                        swarm[liste[neighbor[t]][0]].update(c0,c1,c2,gBest=swarm[tempBest].pos)
        if (typ == 4):
            # Create neighborhoods with 2 particles per neighborhood:
            firstN = numpy.arange(0,int(N/2),dtype=int)
            secondN = numpy.arange(int(N/2),int(N/2)+len(firstN),dtype=int)
            random.shuffle(firstN)          # random shuffeling
            random.shuffle(secondN)         # "
            for i in range(len(firstN)):
                first = firstN[i]
                second = secondN[i]
                if (N%2) and (i==len(firstN)-1):     # N is odd and last loop running
                    third = N-1
                nBest = 0
                if (liste[first][1] <= liste[second][1]):
                    nBest = liste[first][0]
                    if (N%2) and (i==len(firstN)-1) and (liste[third][1]<=liste[first][1]):
                        nBest = liste[third][0]
                else:
                    nBest = liste[second][0]
                    if (N%2) and (i==len(firstN)-1) and (liste[third][1]<=liste[second][1]):
                        nBest = liste[third][0]
                swarm[liste[first][0]].update(c0,c1,c2,gBest=swarm[gBest].pos,nBest=swarm[nBest].pos)     # update velocity and position
                swarm[liste[second][0]].update(c0,c1,c2,gBest=swarm[gBest].pos,nBest=swarm[nBest].pos)
                if (N%2) and (i==len(firstN)-1):
                    swarm[liste[third][0]].update(c0,c1,c2,gBest=swarm[gBest].pos,nBest=swarm[nBest].pos)

        k = k+1
        
        """
        # Untersuchung Verteilung der Partikel:
        verteilung = numpy.zeros([n,3])
        for q in range (n):
            verteilung[q,1] = bounds.lb[q]
            verteilung[q,2] = bounds.ub[q]
            summe = 0
            for p in range(N):
                summe += swarm[p].pos[q]
            middle = summe/N
            maxi = 0
            for p in range(N):
                temp = abs(middle-swarm[p].pos[q])
                if (temp >= maxi):
                    maxi = temp
            verteilung[q,0] = maxi
        print("verteilung = ",verteilung)
        """


def PopBasIncLearning(func,x0,args=(),Ng=100,m=20,**unknown_options):
    # algorithm based on population based incremental learning
    # Np = population size; Ng = number of generations; 
    # m = subintervalls between lower bound and upper bound
    tolError = 1e-3
    stopNum = 10        # when best solution isn't changing significantly after 10 iterations then stop
    n = len(x0)         # problem size
    Np = 3*m          # population size
    #xbest = x0
    #fbest = func(xbest)
    xbest = numpy.empty(n)
    fbest = 1e16
    bounds = unknown_options['bounds']

    # vector delta is needed to define the subintervalls
    delta = numpy.empty(n)
    for i in range(n):
        delta[i] = (bounds.ub[i]-bounds.lb[i])/float(m)

    # initializing probability matrix
    P = numpy.empty([n,m])
    for i in range(n):
        for j in range(m):
            P[i,j] = 1/float(m)

    stopTol = 0
    it = 0
    
    #while it<Ng: 
    while it<Ng:
        # Update population:
        R = [[] for i in range(n)]          # initialization of population list R
        for i in range(n):                  # loop over all components of x
            ctr = 0
            for j in range(m):              # loop over all subintervalls
                for t in range(int(math.floor(Np*P[i,j]))):
                    #rand*(b-a)+a to create a random number betwenn [a,b]
                    a = bounds.lb[i]+j*delta[i]
                    b = bounds.lb[i]+(j+1)*delta[i]
                    R[i].append(random.random()*(b-a)+a)
                    ctr += 1
            while ctr<Np:
                a = bounds.lb[i]
                b = bounds.ub[i]
                R[i].append(random.random()*(b-a)+a)     # wo dazu ordnen?
                ctr += 1
        # random shuffle of all lines in R
        for i in range(n):
            random.shuffle(R[i])
        # determine function values of individuals:
        F = numpy.empty(Np)
        for w in range(Np):
            Ind = numpy.empty(n)
            for i in range(n):
                Ind[i] = R[i][w]                # all x-components of one individual
            F[w] = func(Ind)
            if (F[w]<=fbest):
                if (abs(F[w]-fbest)>tolError):
                    stopTol = 0
                fbest = F[w]
                xbest = Ind
        stopTol += 1    

        print("fbest = ", fbest)

        # find out in which interval xbest is:
        r = numpy.empty(n)          # safes the interval number j of xbest[i]
        for i in range(n):
            for j in range(m):
                if(xbest[i]>=bounds.lb[i]+j*delta[i] and \
                                xbest[i]<=bounds.lb[i]+(j+1)*delta[i]):
                    r[i] = j
                    break
        
        # Updating der Probability Matrix:
        P_s = numpy.empty([n,m])
        for i in range(n):
            for j in range(m):
                L = 0.5*math.exp(-(j-r[i])**2)      # Learning rate
                P_s[i,j] = (1-L)*P[i,j]+L
        # Normalizing each row of P_s -> new P
        for i in range(n):
            summe = 0
            for j in range(m):
                summe += P_s[i,j]
            for j in range(m):
                P[i,j] = (1/summe) * P_s[i,j]

        it += 1
                

    return OptimizeResult(fun=fbest, x=xbest, nit=it+1)


def PopBasIncLearning_hyb(func,x0,args=(),Ng=100,m=20,Q=3,typ=2,**unknown_options):
    # algorithm based on population based incremental learning
    # typ 1: Gradient is the approximative gradient
    # typ 2: Gradient is numerical computet
    # Np = population size; Ng = number of generations; 
    # m = subintervalls between lower bound and upper bound
    #---------------------------------------------------------------------------
    def approx_grad(swarm,func,bounds):
        # in swarm are the best and the next Q best particles (its a list with particle objects)

        A = numpy.empty([Q,n])
        b = numpy.empty(Q)
        xbest = swarm[0].pos
        fbest = swarm[0].f
        
        if typ == 1:                # approximative gradient
            for i in range(Q):
                for j in range(n):
                    xi = swarm[i+1].pos
                    A[i,j] = xbest[j] - xi[j]
                fi = swarm[i+1].f
                b[i] = fbest - fi
    
            gradient = numpy.linalg.pinv(A).dot(b)
            s = -gradient
            #print("gradient = ", s)
        else:                       # numerical gradient    
            h = math.sqrt(numpy.finfo(float).eps)
            s = -grad(func,xbest,h)

        NewPart = []        # list for new particle objects

        check = 0
        beta1 = random.uniform(0,1)     # start value for line-search
        Pos = line_search_bound(func,beta1,xbest,s,bounds)
        if (numpy.all(Pos>=bounds.lb) and numpy.all(Pos<=bounds.ub)):
            NewPart.append(particle(Pos))
        else:
            check = 1
        beta2 = random.uniform(0,1)     # start value for line-search
        Pos = line_search_bound(func,beta2,xbest,s,bounds)
        if (numpy.all(Pos>=bounds.lb) and numpy.all(Pos<=bounds.ub)):
            NewPart.append(particle(Pos))
        else:
            check = 1

        if (check > 0):             # if one of the first two points isn't in feasible region -> stop
            #print("nicht in feasible region")
            return NewPart
        else:
            # solve LGS T*u=w
            f1 = NewPart[0].f
            f2 = NewPart[1].f
            T = numpy.array([[0,0,1],[beta1**2,beta1,1],[beta2**2,beta2,1]])
            w = numpy.array([fbest,f1,f2])
            u = numpy.linalg.solve(T, w)
            
            # determine z3:
            beta3 = -u[1]/2/u[0]
            Pos = xbest + beta3*s
            if (numpy.all(Pos>=bounds.lb) and numpy.all(Pos<=bounds.ub)):
                NewPart.append(particle(Pos))
            else:
                #print("Z3 nicht in feasible region")
                pass
            return NewPart

    def EDR(X,t,Ng):          # Evolutionary Direction Recombination
        # X: list with three random particles
        # t: number of the actual generation (t start from 1)
        # Ng: number of generations (maximum)
        #-----------------------------------------------------------------------
        def SL(t):
            # computes the maximal Step Length
            return math.exp((6.2146/(Ng-1))-0.6931)*math.exp(-(6.2146/(Ng-1))*t)
            
        #-----------------------------------------------------------------------
        n = len(X[0].pos)
        
        # sort particles from f_low to f_high:
        X.sort(key=lambda x: x.f)

        # compute evolutionary direction:
        a = numpy.random.normal(loc=0.0, scale=1.0, size=n)
        c = 0.05*a
        s = (X[0].pos-X[1].pos)+(X[0].pos-X[2].pos)+c

        # calculate new positions:
        NewPart = []
        Pos = X[0].pos + numpy.random.uniform()*SL(t)*s
        if (numpy.all(Pos>=bounds.lb) and numpy.all(Pos<=bounds.ub)):
            NewPart.append(particle(Pos))
        Pos = X[0].pos - numpy.random.uniform()*SL(t)*s
        if (numpy.all(Pos>=bounds.lb) and numpy.all(Pos<=bounds.ub)):
            NewPart.append(particle(Pos))

        return NewPart

    class particle:
        def __init__(self,x0):
            self.pos = x0
            self.f = func(x0)


    #---------------------------------------------------------------------------
    tolError = 1e-3
    stopNum = 10        # when best solution isn't changing significantly after 10 iterations then stop
    n = len(x0)         # problem size
    Np = 200            # population size
    #xbest = x0
    #fbest = func(xbest)
    xbest = numpy.empty(n)
    fbest = 1e16
    bounds = unknown_options['bounds']

    # vector delta is needed to define the subintervalls
    delta = numpy.empty(n)
    for i in range(n):
        delta[i] = (bounds.ub[i]-bounds.lb[i])/float(m)

    # initializing probability matrix
    P = numpy.empty([n,m])
    for i in range(n):
        for j in range(m):
            P[i,j] = 1/float(m)

    stopTol = 0
    it = 0
    r_alt = numpy.zeros(n)
    
    #while it<Ng:  
    while it<Ng:
        # Update subpopulation with (Np/2) individuals:
        R = [[] for i in range(n)]          # initialization of population list R
        for i in range(n):                  # loop over all components of x
            ctr = 0
            for j in range(m):              # loop over all subintervalls
                for t in range(int(math.floor(int(Np/2)*P[i,j]))):
                    #rand*(b-a)+a to create a random number betwenn [a,b]
                    a = bounds.lb[i]+j*delta[i]
                    b = bounds.lb[i]+(j+1)*delta[i]
                    R[i].append(random.random()*(b-a)+a)
                    ctr += 1
            while ctr<int(Np/2):
                a = bounds.lb[i]
                b = bounds.ub[i]
                R[i].append(random.random()*(b-a)+a)     # wo dazu ordnen?
                ctr += 1
        
        # random shuffle of all lines in R
        for i in range(n):
            random.shuffle(R[i])
        
        # create particle objects:
        swarm = []
        for w in range(int(Np/2)):
            Ind = numpy.empty(n)
            for i in range(n):
                Ind[i] = R[i][w]
            swarm.append(particle(Ind))

        # sort swarm in order of function values (low to high):
        swarm.sort(key=lambda x: x.f)

        # update xbest:
        if (swarm[0].f <= fbest):
            fbest = swarm[0].f
            xbest = swarm[0].pos

        # Approsimate Gradient Method:
        NewPart = approx_grad(swarm[0:Q+1],func,bounds)

        # Add new particles to swarm:
        swarm.extend(NewPart)

        # sort swarm again (from f_low to f_high):
        swarm.sort(key=lambda x: x.f)

        # Update xbest:
        if (swarm[0].f < fbest):
            fbest = swarm[0].f
            xbest = swarm[0].pos
            print("Verbesserung durch AGM")

        # Evolutionary Direction Opterator:
        while (len(swarm) <= Np-2):
            a = 0
            b = 0
            c = 0
            while (a==b or a==c or b==c):
                a = random.randint(0,len(swarm)-1)
                b = random.randint(0,len(swarm)-1)
                c = random.randint(0,len(swarm)-1)
            X = []
            X.append(swarm[a])
            X.append(swarm[b])
            X.append(swarm[c])
            t = it+1
            NewPart = EDR(X,t,Ng)       # create 2 children particles
            swarm.extend(NewPart)       # add 2 children to swarm

        # sort swarm again (from f_low to f_high):
        swarm.sort(key=lambda x: x.f)

        # Update xbest:
        if (swarm[0].f < fbest):
            fbest = swarm[0].f
            xbest = swarm[0].pos
            print("Verbesserung durch EDO")

        print("fbest = ", fbest)

        
        # find out in which interval xbest is:
        r = numpy.empty(n)          # safes the interval number j of xbest[i]
        for i in range(n):
            for j in range(m):
                if(xbest[i]>=bounds.lb[i]+j*delta[i] and \
                                xbest[i]<=bounds.lb[i]+(j+1)*delta[i]):
                    r[i] = j
                    break
        
        #print("Abstand Intervall xBest_alt zu xBest_neu = ",numpy.linalg.norm(r-r_alt))
        r_alt = r
        #print("P alt = ", P)
        #print("r = ", r)

        # Updating der Probability Matrix:
        P_s = numpy.empty([n,m])
        for i in range(n):
            for j in range(m):
                L = 0.5*math.exp(-(j-r[i])**2)      # Learning rate
                P_s[i,j] = (1-L)*P[i,j]+L
        # Normalizing each row of P_s -> new P
        for i in range(n):
            summe = 0
            for j in range(m):
                summe += P_s[i,j]
            for j in range(m):
                P[i,j] = (1/summe) * P_s[i,j]

        #print("P neu = ",P)
        it += 1
        
        PBestInt = 0
        for i in range(n):
            PBestInt += P[i,int(r[i])]
        #print("Summe P_best = ", PBestInt)
                

    return OptimizeResult(fun=fbest, x=xbest, nit=it+1)
    """
        # loop over all particles:
        for i in range(N):
            f_i_k = func(swarm[i].pos)
            print("f = ",f_i_k)
            if (f_i_k <= swarm[i].best_part_f): # update best position of particle
                swarm[i].best_part_f = f_i_k
                swarm[i].best_part   = swarm[i].pos
            if (f_i_k <= best_swarm_f):         # update best position of swarm
                best_swarm_f = f_i_k
                best_swarm = swarm[i].pos

        # loop over all particles for Updating:
        for i in range(N):
            swarm[i].update(c1,c2,best_swarm)
        
        # Increment Iteration
        k = k+1

    return OptimizeResult(fun=best_swarm_f, x=best_swarm, nit = k)
    """


# from:
# https://github.com/scipy/scipy/blob/v1.3.3/scipy/optimize/optimize.py#L77-L132
"""
class OptimizeResult(dict):
    Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    
"""
# from 
# https://github.com/scipy/scipy/blob/v1.3.3/scipy/optimize/_minimize.py#L42-L626
"""
def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    .
    .
    .
    **Custom minimizers**
    It may be useful to pass a custom minimization method, for example
    when using a frontend to this method such as `scipy.optimize.basinhopping`
    or a different library.  You can simply pass a callable as the ``method``
    parameter.
    
    The callable is called as 
    
    ``method(fun, x0, args, **kwargs, **options)``
    
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `callback`, `hess`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  Also, if
    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
    `fun` returns just the function values and `jac` is converted to a function
    returning the Jacobian.  The method shall return an `OptimizeResult`
    object.
    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method.  You can find an example in the scipy.optimize tutorial.
    .
    .
    .
"""

# custom optimize signature example from:
# https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
"""
def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1,
            maxiter=100, callback=None, **options):
"""
# another from:
# https://github.com/scipy/scipy/blob/v1.3.3/scipy/optimize/optimize.py#L77-L132
"""
def _minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=1e-4, fatol=1e-4, adaptive=False,
                         **unknown_options):
"""
# How they take the boundaries into account.
# from
# class FloatOptimizableVariable(OptimizableVariable):
"""
 import math

 def left_bounded(x):
     return math.fabs(left)*math.log((x - left)/math.fabs(left))

 def left_bounded_inv(x):
     return left + math.fabs(left)*math.exp(x/math.fabs(left))

 def right_bounded(x):
     return -math.fabs(right)*math.log((right - x)/math.fabs(right))

 def right_bounded_inv(x):
     return right - math.fabs(right)*math.exp(-x/math.fabs(right))

 def both_bounded(x):
     return math.log((-x + left)/(x - right)) * math.fabs(left - right)

 def both_bounded_inv(x):
     return left +\
             (right - left)/(1. + math.exp(-x/math.fabs(right - left)))


"""                                                                          
 
