from __future__ import print_function
import math
import random
import numpy
import scipy
import sys
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from pyrateoptics.core.log import BaseLogger
from pyrateoptics.optimize.optimize_backends import Backend
from derivatives import grad, grad_pen, grad_lag, grad_log, hessian
from auxiliary_functions import get_bdry, eval_h, eval_c, my_log, printArray,\
                                termcondition

class ProjectScipyBackend(Backend):
    def __init__(self, optimize_func, methodparam=None, tau0=1.0,
                 options={}, stochagradparam=False, **kwargs):
        self.optimize_func   = optimize_func
        self.options         = options
        self.methodparam     = methodparam
        self.stochagradparam = stochagradparam
        self.kwargs          = kwargs
        self.tauk            = tau0
        self.lamk            = 1.
        self.stochagrad      = None
        # self.func = MeritFunctionWrapper is set within the Optimizer __init__ 

    def update_PSB(self, optimi) :
        '''
        function gets the boundaries for the variables. THis is necessary if 
        you want to run the optimization with penalty/lagrange terms. It is
        important to call this function AFTER the Optimizer object was created
        and BEFORE you run the optimization.
        '''

        self.bdry = get_bdry(optimi)
        printArray('bdry in ProjectScipyBackend = ', self.bdry)
    
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

        elif (self.methodparam == 'penalty'):

            # optimize: meritfunction(x) + 0.5*tau*||h(x)||_2^2

            print('----------------- run penalty -----------------')

            xk_1 = x0
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

                # for benchmark
                self.kwargs['jac'] = grad_total
 
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
                               options=self.options, 
                               **self.kwargs)
                # update xk
                xk = res.x

                printArray('x_k =', res.x, point=points)
                print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))

                # check if xk is in the feasible set with ||h(x)||_inf < 10*eps
                if (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf) < tol_seq):
                    print('\n----------- end of penalty run -----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                else: # xk is not in the feasible set -> update tauk
                    # update tau
                    self.tauk = 7*self.tauk 

                # update xk-1 
                xk_1 = xk


        elif (self.methodparam == 'penalty-lagrange'):

            # optimize: meritfunction(x) + lambda^T h(x) + 0.5*tau*||h(x)||_2^2

            print('----------------- run penalty lagrange -----------------')

            # choose the initial lamda
            self.lamk = 0.333*numpy.ones(2*len(x0))

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
                self.options['grad']= grad_total
            
                # for benchmark
                self.kwargs['jac'] = grad_total

                # update iteration number
                iterNum += 1

                print('\niteration number = %d' % (iterNum))
                printArray('bdry =', self.bdry)
                print('\ntau = %6.3f' % (self.tauk))
                printArray('lambda_k =', self.lamk)
                printArray('gradient of the penalty term =',\
                           grad_lag(xk,self.bdry, self.tauk, self.lamk))


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
                               options=self.options, 
                               **self.kwargs)
                # update xk
                xk = res.x

                printArray('x_k =', res.x, point=points)
                print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))

                # check if xk is in the feasible set with ||h(x)||_inf < 10*eps
                if (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf) < tol_seq):
                    print('\n---------- end of penalty lagrange run ----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
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

                # for benchmark
                self.kwargs['jac'] = grad_total

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
                if (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf) < tol_seq):
                    print('\n---------- end of log barrier run----------')
                    print('\nTerminated in iteration = %d' % (iterNum))
                    print('\n||h(x)||_inf = %5.3f' % \
                          (numpy.linalg.norm(eval_h(xk, self.bdry), numpy.inf)))
                    break
                else: # xk is not in the feasible set -> update my
                    # update my
                    self.my = self.my/10

                # update xk-1
                xk_1 = xk

        else:
            print('Methodparam not found!')
            sys.exit()

        printArray('x_k =', res.x, point=points)
        print('\nmeritfunction(x_k) = %10.6f' % (self.func(res.x)))
        self.res = res
        return res.x


def sgd(func, x0, args, 
        maxiter=500,
        stepsize=1e-9,
        methods='nag',
        gamma=0.9,
        **kwargs):

    gradient = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum = 0
    xk_1 = x0
    fk_1 = func(x0)

    # momentum vector
    vk_1 = numpy.zeros(len(x0))

    while (1):
        # update iteration number
        iternum += 1

        # choose the method
        if (methods == 'vanilla'):
            vk = stepsize*stochagrad(xk_1)
        if (methods == 'momentum'):
            # momentum vector
            vk = gamma*vk_1+stepsize*stochagrad(xk_1)
            vk_1 = vk
        if (methods == 'nag'):
            # nesterov accelerated gradient
            vk = gamma*vk_1+stepsize*stochagrad(xk_1-gamma*vk_1)
            vk_1 = vk

        # iteration rule
        xk = xk_1 - vk 
        # debugging
        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))

        # termination
        if ((iternum >= maxiter) or\
            ((fk < 3.) and (gknorm <= 500))):
            break
        xk_1 = xk
        fk_1 = fk

    return OptimizeResult(fun=fk, x=xk, nit=iternum)


def adam(func, x0, args, 
         maxiter=300,
         stepsize=1e-2,
         beta1=0.1,
         beta2=0.99,
         epsilon=1e-2,
         **kwargs):

    gradient = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum = 0
    xk_1 = x0
    fk_1 = func(x0)

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
        # iteration rule
        xk = xk_1 - stepsize*mk_hat/(numpy.power(vk_hat,0.5)+epsilon)

        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))
        # termination
        if (fk < 1e-10):
            break

        if ((iternum >= maxiter) or\
            ((fk < 3.) and (gknorm <= 500))):
            break
        xk_1 = xk
        fk_1 = fk

    return OptimizeResult(fun=fk, x=xk, nit=iternum)


def adamax(func, x0, args, 
           maxiter=300,
           stepsize=1e-2,
           beta1=0.09,
           beta2=0.99,
           thetaf=1e-1,
           p=None,
           **kwargs):

    gradient = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum = 0
    xk_1 = x0
    fk_1 = func(x0)
    ismin = False

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
        # termination
        if (fk < 1e-10):
            break

        if ((iternum >= maxiter) or\
            ((fk < 3.) or (gknorm <= 500))):
            break
        xk_1 = xk
        fk_1 = fk

    return OptimizeResult(fun=fk, x=xk, nit=iternum)


def adagrad(func, x0, args, 
            maxiter=300,
            stepsize=1e-3,
            epsilon=1e-3,
            **kwargs):

    gradient = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum = 0
    dim = len(x0)
    xk_1 = x0
    fk_1 = func(x0)
    G = numpy.zeros(dim)

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

        # termination
        if (fk < 1e-10):
            break

        if ((iternum >= maxiter) or\
            ((fk < 3.) and (gknorm <= 500))):
            break
        xk_1 = xk
        fk_1 = fk

    return OptimizeResult(fun=fk, x=xk, nit=iternum)

def adadelta(func, x0, args, 
             maxiter=300,
             stepsize=1e-3,
             gamma=0.9,
             epsilon=1e-3,
             **kwargs):

    gradient = kwargs['grad']
    stochagrad = kwargs['stochagrad']
    iternum = 0
    dim = len(x0)
    xk_1 = x0
    fk_1 = func(x0)
    gk = stochagrad(x0)
    Ek_1 = numpy.zeros(dim)

    while (1):
        # update iteration number
        iternum += 1
        # get stochastic gradient
        gk = stochagrad(xk_1)
        # update E
        Ek   = gamma*Ek_1+(1-gamma)*gk*gk
        Ek_1 = Ek
        # delta
        deltak = -stepsize
        # iteration rule
        xk = xk_1 - stepsize*(gk/(numpy.sqrt(G)+epsilon))

        # debugging
        fk = func(xk)
        gk = gradient(xk)
        gknorm = numpy.linalg.norm(gk, numpy.inf)
        print('gknorm = %7.4f' %(gknorm))
        print('fk     = %7.4f' %(fk))

        # termination
        if (fk < 1e-10):
            break

        if ((iternum >= maxiter) or\
            ((fk < 3.) and (gknorm <= 500))):
            break
        xk_1 = xk
        fk_1 = fk

    return OptimizeResult(fun=fk, x=xk, nit=iternum)



def gradient_descent(func, x0, args=(),
                     maxiter=100, stepsize=1e-8, **kwargs):
    xk = x0
    grad = kwargs['grad']
    iternum = 0 

    while (iternum < maxiter):
        iternum += 1
        xk -= stepsize*grad(xk)
        if (numpy.linalg.norm(grad(xk), numpy.inf) < 1e-1):
            print('\ngradient is near zero\n')
            print('||grad(x_final)||_inf = %10.6'\
                  % (numpy.linalg.norm(grad(xk), numpy.inf)))
            print('\nTerminated in iteration: %d' % (iternum))
            break

    printArray('gradient in gradient decsent for x_final = ', grad(xk))
    return OptimizeResult(fun=func(xk), x=xk, nit=iternum)

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
 
