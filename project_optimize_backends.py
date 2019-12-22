import math
import random
import numpy
import sys
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from pyrateoptics.core.log import BaseLogger
from pyrateoptics.optimize.optimize_backends import Backend
from derivatives import grad, grad_pen, grad_lag, grad_log
from auxiliary_functions import get_bdry, eval_h, eval_c, my_log

class ProjectScipyBackend(Backend):
    def __init__(self, optimize_func, methodparam='standard', tau0=0.0,
                 options={}, **kwargs):
        self.optimize_func = optimize_func
        self.options       = options
        self.methodparam   = methodparam
        self.kwargs        = kwargs
        self.tauk          = tau0
        self.lamk          = 0.
        # self.func=MeritFunctionWrapper is set within the Optimizer __init__ 

    def update_PSB(self, optimi) :
        '''
        function gets the boundaries for the variables. THis is necessary if 
        you want to run the optimization with penalty/lagrange terms. It is
        important to call this function AFTER the Optimizer object was created
        and BEFORE you run the optimization.
        '''

        self.bdry = get_bdry(optimi)
        print('self bdry in PSB:')
        print(self.bdry)
    
    def run(self, x0):

        tol_seq = 0.01  #tolerance for sequence of optimization (pen/lag/log)

        if (self.methodparam == 'standard'):
            print('----------------- run standard -----------------')
            print('\nx0 =')
            print(x0)

            #No Penalty, no Lagrange terms
            res = minimize(self.func, 
                           x0=x0, 
                           args=(), 
                           method=self.optimize_func,
                           options=self.options, 
                           **self.kwargs)
            
        elif (self.methodparam == 'penalty'):
            #Penalty term, but no Lagrange term
            print('----------------- run penalty -----------------')

            ########DEFINE GRADIENT###########
            def grad_total(x):
                h = 0.0000001
                res = grad(self.func, x, h) +\
                            grad_pen(x,self.bdry,self.tau0)
                return res
            
            self.options['grad'] = grad_total
            ########

            # for benchmark
            self.kwargs['jac'] = grad_total
            
            #################Iteration over different Tau's
            xk_1 = x0
            xk   = x0
            for i in range(15): #max amount of different tau's
                print('\nx_k =')
                print(xk)
                print('\nbdry =')
                print(self.bdry)
                print('\ntau =')
                print(self.tauk)
                print('\ngradient of the penalty term')
                print(grad_pen(xk,self.bdry, self.tauk))
                res = minimize(lambda x: self.func(x) +
                                        0.5*self.tauk*numpy.square(
                                        numpy.linalg.norm(eval_h(x,self.bdry))), 
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               options=self.options, 
                               **self.kwargs)
                xk = res.x
                #ABBRUCHBEDINGUNG, wie waehlt man tol?!?!?!?!?!
                print('\nmeritwert=')
                print(self.func(xk))
                if numpy.absolute(numpy.linalg.norm(xk-xk_1))<tol_seq :
                    print('\n\n\
                           ***ABBRUCH DER OPTIMIERUNGSSEQUENZ***\n\n')
                    print(i)
                    break

                xk_1 = xk  # update resalt
                
                self.tauk = 7*self.tauk #update tau
            ##############
        
        
        elif (self.methodparam == 'penalty-lagrange'):
            #Both, Penalty and Lagrange Term
            print('----------------- run penalty lagrange -----------------')

            self.lamk = 0.333*numpy.ones(2*len(x0))
            ########DEFINE GRADIENT###########
            def grad_total(x):
                h = 0.0000001
                res = grad(self.func, x, h) +\
                            grad_lag(x,self.bdry,self.tauk,self.lamk)
                return res
            
            self.options['grad']= grad_total
            ########
            
            # for benchmark
            self.kwargs['jac'] = grad_total

            #################Iteration over different Tau's
            xk_1 = x0
            xk   = x0
            for i in range(15): #max amount of different tau's
                print('\nx_k =')
                print(xk)
                print('\nbdry =')
                print(self.bdry)
                print('\ntau_k =')
                print(self.tauk)
                print('\nlambda_k =')
                print(self.lamk)
                print('\ngradient of the penalty term')
                print(grad_pen(xk,self.bdry, self.tauk))
                res = minimize(lambda x: self.func(x) +
                                        0.5*self.tauk*numpy.square(
                                        numpy.linalg.norm(eval_h(x,self.bdry)))
                                        + 
                                        numpy.dot(self.lamk,eval_h(x,self.bdry)),
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               options=self.options, 
                               **self.kwargs)
                xk = res.x
                #ABBRUCHBEDINGUNG, wie waehlt man tol?!?!?!?!?!
                print('meritwert')
                print(self.func(xk))
                if numpy.absolute(numpy.linalg.norm(xk-xk_1))<tol_seq :
                    print('\n\n\
                           ***ABBRUCH DER OPTIMIERUNGSSEQUENZ***\n\n')
                    print(i)
                    break

                xk_1 = xk                               #update resalt
                self.lamk = numpy.add(self.lamk, self.tauk*eval_h(xk, self.bdry))
                                                        #update lam
                self.tauk = 7*self.tauk                 #update tau
            ##############
        
        elif (self.methodparam == 'log'):
            # Logarithmic Barrier Method
            print('----------------- run log barrier -----------------')

            self.my = 1.0 # '.0' is important, otherwise its an integer and my=0 
                          # in second step!
            ########DEFINE GRADIENT###########
            def grad_total(x):
                h = 0.0000001
                res = grad(self.func, x, h) -\
                           grad_log(x,self.bdry,self.my)
                return res
            
            self.options['grad']= grad_total
            ########

            # for benchmark
            self.kwargs['jac'] = grad_total

            #################Iteration over different my's
            xk_1 = x0
            xk   = x0
            for i in range(15): #max amount of different my's
                print('\nxk =')
                print(xk)
                res = minimize(lambda x: self.func(x) - self.my*numpy.sum(
                                         my_log(eval_c(x,self.bdry))),
                               x0=xk, 
                               args=(), 
                               method=self.optimize_func,
                               options=self.options, 
                               **self.kwargs)
                xk = res.x
                #ABBRUCHBEDINGUNG, wie waehlt man tol?!?!?!?!?!
                normneu = numpy.linalg.norm(xk)
                print('\nmeritwert')
                print(self.func(xk))
                if numpy.absolute(numpy.linalg.norm(xk-xk_1))<tol_seq :
                    print('\n\n\
                           ***ABBRUCH DER OPTIMIERUNGSSEQUENZ***\n\n')
                    print(i)
                    break

                xk_1 = xk                          #update resalt

                self.my = self.my/10               #update my
            ##############

        else:
            print('Methodparam not found!')
            sys.exit()

        self.res = res
        return res.x


def sgd(func, x0, args, 
        gradient=None, maxiter=250, fatol=1e-4,
        stepsize=1e-3, h=1e-6, **unknown_options):
    """
    maxiter: int
        Maximum allowed number of iterations.
    disp : bool
        Set to True to print convergence messages.
    fatol : number
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    stepsize: number
        Size of step between iterations.
    """
    if (gradient == None):
        print("stochastic_grad is None")
        gradient = grad
    else:
        print("stochastic_grad is not None")
    
    print(x0)
    termcond = grad
    xopt = x0
    xtemp = x0
    # path = numpy.array([x0])
    iterNum = 1
    iterTolf = fatol + 1
    # as a termination condition one could think of the real gradient and the
    # hessian matrix
    # another extension could be the escaping of a local minimum, like:
    # 1. detect the lokal minimum with the gradient and the hessian
    # 2. escape by making more steps with the stochastic gradient until another
    #    local minimum is reached
    tol = 1e-1
    while ((iterNum < maxiter) and (tol < numpy.linalg.norm(termcond(func,xopt,h)))):# and (fatol < iterTolf)):
        # iteration rule
        xopt -= stepsize*gradient(func, xopt, h=1e-6) 

        # print(xopt)

        # append the value to the path
        # numpy.append(path, [xopt], axis=0)
        # update the termination condition
        print(iterNum)
        iterNum += 1
        # -----------------debugging
        iterTolf = numpy.absolute(func(xopt)-func(xtemp))
        # update xtemp
        xtemp = xopt

    return OptimizeResult(fun=func(xopt), x=xopt, nit=iterNum)


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
 
