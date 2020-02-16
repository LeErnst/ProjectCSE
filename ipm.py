from __future__ import print_function
import math
import random
import numpy
import scipy
from scipy import optimize
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
                                termcondition, plot2d, plot3d

from numpy.linalg import inv 

def basicipm(func, x, args,
            reduced=False,
            maxiter=50,
            stepsize=1e-7,  # for Hesse Matrix/Gradient
            gradtol=1,
            **kwargs):
    '''
    Basic interior point method for nonconvex functions
    '''
    m = 2*len(kwargs['bounds'].lb)
    bdry = numpy.zeros(m)
    for i in range(m/2):
        bdry[2*i]   = kwargs['bounds'].lb[i]
        bdry[2*i+1] = kwargs['bounds'].ub[i]
    print(bdry)

    niter = 0
    n     = len(x)
    #starting parameter
    z     = numpy.ones(m)
    s     = numpy.ones(m)
    my    = 20.0
    delta_old = 0
    #textdummy = open("DUMMY3.txt", "a")
    gradtol = 0.000000001
    #setup A
    A = numpy.zeros((m,n))
    for i in range(n) :
        A[2*i,i]   =  1.0
        A[2*i+1,i] = -1.0

    #c_x = eval_c(x, bdry)
    #temp = 20
    #while (numpy.linalg.norm(grad(func, x, stepsize))>gradtol and niter<maxiter) :
    while niter<3 :
        niter = niter + 1
        #print('NITER=')
        #print(niter)
        #error = ipm_error(func, x, A, c_x, z, s, my, textdummy)
        #erroralt = 0
        testit = 0
        #while (error > 30*my and testit<22) :
        while (testit<22) :
            testit = testit + 1
            #erroralt = error
            print('INERTIA CORRECTION....')
            delta, gamma = inertia_correction_ldl(setup_Jac_ipm, func, A, x, z, s, my, delta_old, a=1, b=0.5)
            #delta =1
            print('INERTIA CORRETION Done')
            print('DELTA=')
            print(delta)
            print('WAITING FOR NEWTON...')
            if reduced :
                x,s,z = solve_ipm_newton(setup_F_ipm_reduced, setup_Jac_ipm_reduced, func, A, x, z, s, my, bdry, delta, delta_old, reduced=reduced, variant='gmres')
            else :
                x,s,z = solve_ipm_newton(setup_F_ipm, setup_Jac_ipm, func, A, x, z, s, my, bdry, delta, delta_old, reduced=reduced, variant='gmres')
            #error = ipm_error(func, x, A, c_x, z, s, my, textdummy)
            delta_old = delta
            wertwert= func(x)
        #my = my/2
        my = update_my(s,z)
        print('#################################################################')
        print('NEW MY IS')
        print(my)
    result = OptimizeResult(fun=func, x=x, nit=niter)
    print('gradient=')
    print(numpy.linalg.norm(grad(func, x, stepsize)))
    print('X=')
    print(x)
    print('S=')
    print(s)
    print('Z=')
    print(z)
    
    return result

def update_my(s,z) :
    '''
    updates my according to nocedal p. 572/573
    '''
    m     = len(s)
    p     = numpy.dot(s,z)/m
    temp  = numpy.min(numpy.multiply(s,z))
    tt    = temp/p
    t2    = numpy.minimum(0.05*(1-tt)/tt, 2)
    sigma = 0.1*t2*t2*t2

    return sigma*p


def solve_ipm_newton(F, DF, f, A, x, z, s, my, bdry, delta, delta_old, reduced=False, variant='standard', stepsize=1e-9) :
    '''
    solves the nonlinear system (Nocedal p. 569) with a naiv Newton method
    delta is the factor of the identity matrix that is added on DF to ensure
    the matrix is positiv definit
    variant for netwon can be:
    - 'standard': naiv computation of DF^-1*f

    - 'ldl': computes the LDL' ecomposation and solves the system via Vorwaerts/
             Rueckwaertseinsetzen

    - 'gmres': uses the scipy generalized minimal residual iteration
    
    '''
    n = len(x)
    m = len(z)
    tol = 0.00000000000001
    maxit = 1#7#20#300
    nit   = 0
    fval  = F(f, A, x, z, s, my, bdry)

    text = open("AuswertungIPM.txt", "a")
    #text.write("error1 error2 error3 NewtonIter delta meritwert error Fval gradMerit my\r\n")
    dummy = 1
    while(1) :

        merwertneu=f(x)
        ####################TEXTFILE#####################################
        c_x = eval_c(x, bdry)
        error = ipm_error(f, x, A, c_x, z, s, my, text)
        merwert=merwertneu
        print('MERITWERT IN NEWTON=')
        print(merwert)
        text.write("%d %.4f %.4f %.4f %.4f %.4f %.4f\r\n" % (nit, delta, merwert, error, numpy.linalg.norm(fval), numpy.linalg.norm(grad(f,x,stepsize)), my))    

        #################################################################
        #print('FVAL IN NEWTON')
        #print(fval)
        #fval  = F(f, A, x, z, s, my, bdry)
        if numpy.linalg.norm(fval) < tol or nit>=maxit :
            break
        
        nit = nit + 1
        print('Newton Iteration =')
        print(nit)
        dfval = DF(f, A, x, z, s, delta=delta)
        if variant== 'standard' :
            dx_ges= numpy.dot(numpy.linalg.inv(dfval),fval)
        elif variant== 'ldl' :
            lu, d, per = scipy.linalg.ldl(dfval)
            dx_ges = solve_ldlxb(lu,d,per,fval)
        elif variant== 'gmres':
            dfvals = scipy.sparse.csr_matrix(dfval)
            dx_ges, exitcode= scipy.sparse.linalg.gmres(dfvals, fval)
        
        if reduced :
            sigma = numpy.dot(numpy.diag(numpy.reciprocal(s)), numpy.diag(z))
            siginv= numpy.dot(numpy.diag(numpy.reciprocal(z)), numpy.diag(s))
            
            temp  = numpy.dot(sigma, numpy.diag(numpy.reciprocal(z)))
            temp2 = numpy.dot(sigma,A)

            pz = my*numpy.dot(temp,numpy.ones(m)) - numpy.dot(temp2,dx_ges) - numpy.dot(sigma, eval_c(x,bdry))
            ps = my*numpy.dot(numpy.diag(numpy.reciprocal(z)),numpy.ones(m))
            ps = ps - numpy.dot(siginv, pz) - s

            als, alz = get_alphas(s, z, ps, pz)
            #x,s= filter_meth(f, x , als, dx_ges, s, ps, bdry, my) 
            
            #if dummy and merwertneu<0.31 :
            #    dummy = 0

            if (1) : #(my==20.0) :
                x  = x + als * dx_ges
                s  = s + als * ps
                z  = z + alz * pz
            else :
                print('SCHRITTWEITE ANPASSEN!!!!!!!!!')
                for i in range(10) :
                    x  = x + (als/(i+1)) * dx_ges
                    s  = s + (als/(i+1)) * ps
                    z  = z + (alz/(i+1)) * pz
                    merwertneu = f(x)
                    if merwertneu<merwert :
                        break
                
            fval  = F(f, A, x, z, s, my, bdry)
        else :
            ps    = dx_ges[n:n+m]
            pz    = dx_ges[n+m:n+m+m]
            als, alz = get_alphas(s, z, ps, pz)
            x = x + als * dx_ges[0:n]
            s = s + als * ps
            z = z + alz * pz

        fvalt = fval
        fval  = F(f, A, x, z, s, my, bdry)
        #tei   = 2
        #prevent the solution to get worse
        #while numpy.linalg.norm(fval)-numpy.linalg.norm(fvalt)>10 :
        #    print('WERT HAETTE SICH VERSCHLECHTERT')
        #    x = x - (als/tei) * dx_ges[0:n]
        #    s = s - (als/tei) * ps
        #    z = z + (alz/tei) * pz
        #    tei = tei*2
    return x, s, z

def filter_meth(f, x , als, dx, s, ps, bdry, my):
    '''
    filters the stepsize according to Waechter/Biegler p. 31
    '''
    delta    = 0.1
    stheta   = 2.0
    sphi     = 2.2
    thetamin = 10000
    eta      = 0.02
    gammath  = 0.1
    gammaph  = 0.1

    gradphi  = numpy.dot(grad(f,x,1e-8), dx) 
    xneu     = x + als*dx
    sneu     = s + als*ps
    phi_neu  = f(xneu) - my*numpy.sum(numpy.log(sneu))
    phi      = f(x)     - my*numpy.sum(numpy.log(s))
    theta    = numpy.linalg.norm(eval_c(x, bdry) - s)
    thetaneu = numpy.linalg.norm(eval_c(xneu, bdry) - sneu)
    
    if theta <= thetamin and gradphi < 0 and als* ((-1)*gradphi)**sphi < delta* theta**stheta :
        #armijo
        if phi_neu <= phi + eta*als*gradphi :
            # Case I
            print('CASE I')
            print(als)
            return xneu, sneu
    else :
        print('Case II')
        if thetaneu<= (1-gammath)*theta or phi_neu <= phi - gammaph*theta :
            return xneu, sneu
    print('FAILFAILFAILFAILFIAL')
    return 0,0


def inertia_correction_gersh(A) :
    '''
    delivers a delta that needs to be added to the diagonal of A to ensure that
    all gershgorin circles are inside the positiv area, which means that A is 
    then positiv definit (1 is added to prevent SEMI-positive definite)
    '''

    delta = 0
    n     = A.shape[0]
    for i in range(n) :
        s = numpy.sum(numpy.absolute(A[i,0:n])) - numpy.absolute(A[i,i])
        diff = A[i,i]-s
        if diff<=0 :
            delta = numpy.maximum(delta, numpy.absolute(diff)) 

    return delta+1


def solve_Lxb(L, b) :
    '''
    solves L*x=b, Vorwaertseinsetzen
    '''
    n = len(b)
    x = numpy.zeros(n)
    for i in range(n) :
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i,j]*x[j]
        x[i] = x[i]/L[i,i]
    
    return x

def solve_Rxb(R,b):
    '''
    solves R*x=b, Rueckwaertsseinsetzen
    '''
    n = len(b)
    x = numpy.zeros(n)
    for i in range(n-1,-1,-1) :
        x[i] = b[i]
        for j in range(n-1,i,-1):
            x[i] = x[i] - R[i,j]*x[j]
        x[i] = x[i]/R[i,i]
    
    return x


def solve_Dxb(D,b):
    '''
    solves Dx=b with D being a bolckdiagonal matrix
    '''
    n = len(b)
    x = numpy.zeros(n)
    i = 0
    while (1) :
        if i>=n-1:
            break
        if D[i,i+1] == 0 :
            x[i] = b[i]/D[i,i]
        else:
            x[i:i+2] = numpy.dot(numpy.linalg.inv(D[i:i+2,i:i+2]),b[i:i+2])
            i = i+1
        i = i+1
    
    if x[n-1]==0 :
        x[n-1] = b[n-1]/D[n-1,n-1]

    return x


def solve_ldlxb(lu,d,p,fval):
    '''
    Solves the equation
    '''
    lu = lu[p,:]
    print('LXB')
    y1 = solve_Lxb(lu,fval[p])
    print('DXB')
    y2 = solve_Dxb(d,y1)
    print('RXB')
    x  = solve_Rxb(numpy.transpose(lu),y2)
    print('DOne solve')

    x = x[p]
    print('VOR RETURN SOLVE LDLCB') 
    return x


def eig_blockdia(D) : 
    '''
    returns the eigenvalues of A blockdiagonal matrix D
    '''
    n   = D.shape[0]
    res = numpy.zeros(n)
    i=0
    while(1) :
        if i>=n-1 :
            break
        if D[i,i+1]==0 :
            res[i] = D[i,i]
        else :
            w, v = numpy.linalg.eig(D[i:i+2, i:i+2])
            res[i:i+2] = w
            i = i+1
        i = i+1
    if res[n-1]== 0 :
        res[n-1] = D[n-1,n-1]
        
    return res


def inertia_correction_ldl(J, f, A, x, z, s, my, delta_old, a=1, b=0.5) :
    '''
    algorith B.1 Nocedal
    with LDL factorization to compute eigenvalues
    '''
    
    n = len(x)
    m = len(z)
    Jac = J(f, A, x, z, s, delta=delta_old)
    lu, d, per = scipy.linalg.ldl(Jac)
    w   = eig_blockdia(d)
    pos = amount_positive(w)
    neg = amount_negative(w)
    zer = len(w)-pos-neg

    if (n+m==pos and m==neg and zer==0) :
        delta = 0
        gamma = 0
        return delta, gamma
    if (zer!=0) :
        gamma = 1.0e-8*a*numpy.power(my,b)
    if (delta_old==0) :
        delta = 1.0e-4
    else :
        delta = delta_old/2
    
    gamma = 0 #this is a dummy as we do not need gamma for our project

    cnt = 0
    while(1) :
        cnt = cnt+1
        # TODO: statt immer die jacobi neu zu berechnen, auf die alte einfach delta aufaddieren?!?
        Jac[0:n,0:n]   = Jac[0:n,0:n]+delta*numpy.eye(n)
        #Jac = J(f, A, x, z, s, delta=delta)
        lu, d, per = scipy.linalg.ldl(Jac)
        w   = eig_blockdia(d)
        pos = amount_positive(w)
        neg = amount_negative(w)
        zer = len(w)-pos-neg
        if (n+m==pos and m==neg and zer==0) :
            #delta_alt = delta not necessary as I do this inside the basicipm function
            print('DELTA AND GAMMA SEARCH SUCCESFUL')
            return delta, gamma
        elif cnt>10000 :
            print('ACHTUNG!!!MATRIX KONNTE NICHT POSITIV DEFINIT GEMACHT WERDEN')
            return delta,gamma
        else:
            #print('NICHT AKZEPTIERTES DELTA IST')
            #print(delta)
            delta = delta*10

def get_alphas(s, z, ps, pz, tau=0.995) :
    '''
    Fraction to the boundary rule according to Nocedal p. 567
    '''
    m = len(z)
    alphas = 1.0
    alphaz = 1.0
    nit    = 0
    maxit  = 500
    while(1) :
        if numpy.min(s+alphas*ps-(1-tau)*s) >= 0 or nit>maxit :
            break
        else :
            alphas = alphas/2
            nit = nit +1
    it = 0
    while(1) :
        if numpy.min(z+alphaz*pz-(1-tau)*z) >= 0 or nit>maxit :
            break
        else :
            alphaz = alphaz/2
            nit = nit +1
    
    return alphas, alphaz
        

def inertia_correction(J, f, A, x, z, s, my, delta_old, a=1, b=0.5) :
    '''
    algorith B.1 Nocedal
    desperately needs to be programmed more effective, as right now it ruins 
    the runtime as it calculates a lot of Hesseian matrices plut their
    eigenvalues
    '''
    
    n = len(x)
    m = len(z)
    Jac = J(f, A, x, z, s, delta=delta_old)
    w,v = numpy.linalg.eig(Jac)
    pos = amount_positive(w)
    neg = amount_negative(w)
    zer = len(w)-pos-neg

    if (n+m==pos and m==neg and zer==0) :
        delta = 0
        gamma = 0
        return delta, gamma
    if (zer!=0) :
        gamma = 1.0e-8*a*numpy.power(my,b)
    if (delta_old==0) :
        delta = 1.0e-4
    else :
        delta = delta_old/2
    
    gamma = 0 #this is a dummy as we do not need gamma for our project

    cnt = 0
    while(1) :
        cnt = cnt+1
        Jac = J(f, A, x, z, s, delta=delta)
        w,v = numpy.linalg.eig(Jac)
        pos = amount_positive(w)
        neg = amount_negative(w)
        zer = len(w)-pos-neg
        if (n+m==pos and m==neg and zer==0) :
            #delta_alt = delta not necessary as I do this inside the basicipm function
            print('DELTA AND GAMMA SEARCH SUCCESFUL')
            return delta, gamma
        elif cnt>10000 :
            print('ACHTUNG!!!MATRIX KONNTE NICHT POSITIV DEFINIT GEMACHT WERDEN')
            return delta,gamma
        else:
            delta = delta*100



def amount_positive(w) :
    '''
    counts number of elements >0 in array
    '''
    cnt = 0
    for i in w :
        if (i>0) :
            cnt = cnt + 1

    return cnt


def amount_negative(w) :
    '''
    counts number of elements <0 in array
    '''
    cnt = 0
    for i in w :
        if (i<0) :
            cnt = cnt + 1

    return cnt
    

def setup_F_ipm(f, A, x, z, s, my, bdry, stepsize=1e-9):
    '''
    Function to set up RHS for ipm (Nocedal p. 569)
    '''

    n = len(x)
    m = len(z)
    F = numpy.zeros(n+m+m)

    F[0:n]       = numpy.subtract(grad(f,x,stepsize), numpy.dot(numpy.transpose(A),z))
    F[n:n+m]     = numpy.subtract(z, my*numpy.dot(numpy.diag(numpy.reciprocal(s)),numpy.ones(m)))
    F[n+m:n+m+m] = numpy.subtract(eval_c(x, bdry), s)

    return -1*F

def setup_F_ipm_reduced(f, A, x, z, s, my, bdry, stepsize=1e-9):
    '''
    Function to set up RHS for ipm for the reduced system(Nocedal p. 571)
    '''
    n = len(x)
    m = len(bdry)
    F = numpy.zeros(n)
    sigma = numpy.dot(numpy.diag(numpy.reciprocal(s)), numpy.diag(z))

    F[0:n]       = numpy.subtract(grad(f,x,stepsize), numpy.dot(numpy.transpose(A),z))
    temp   = numpy.dot(numpy.transpose(A),sigma)
    temp   = numpy.dot(temp, numpy.diag(numpy.reciprocal(z)))
    temp2  = numpy.dot(numpy.transpose(A), sigma)
    F[0:n] = F[0:n] - my*numpy.dot(temp,numpy.ones(m)) + numpy.dot(temp2,eval_c(x,bdry))

    return -1*F

def setup_Jac_ipm(f, A, x, z, s, delta=0, stepsize=1e-9):
    '''
    Function to set up Jacobian of F (Nocedal p. 567 bzw. p.574 with gamma/delta
    to ensure that matrix is pos. definite)
    '''
    
    n = len(x)
    m = len(z)
    J = numpy.zeros((n+m+m,n+m+m))
    
    # First column
    J[0:n,       0:n]   = hessian(f, x)+delta*numpy.eye(n)
    J[n+m:n+m+m, 0:n]   = A

    #second column
    J[n:n+m,    n:n+m]  = numpy.dot(numpy.diag(numpy.reciprocal(s)), numpy.diag(z))
    J[n+m:n+m+m,n:n+m]  = -1*numpy.eye(m)

    #third column
    J[0:n,   n+m:n+m+m] = numpy.transpose(A)
    J[n:n+m, n+m:n+m+m] = -1*numpy.eye(m)

    return J

def setup_Jac_ipm_reduced(f, A, x, z, s, delta=0):
    '''
    Function to set up Jacobian of F_reuced (Nocedal p. 571 with delta to ensure
    that matrix is pos. definite)
    '''
    n = len(x)
    J = numpy.zeros((n,n))
    sigma = numpy.dot(numpy.diag(numpy.reciprocal(s)), numpy.diag(z))
    temp = numpy.dot(numpy.transpose(A),sigma)
    J = hessian(f, x) + delta*numpy.eye(n) + numpy.dot(temp,A)

    return J
            
def ipm_error(f, x, A, c_x, z, s, my, text, stepsize=1e-9):
    a = numpy.subtract(grad(f,x,stepsize),numpy.dot(numpy.transpose(A), z))
    #print('INSIDE ERROR')
    #print('grad =')
    #print(numpy.linalg.norm(grad(f,x,stepsize)))
    
    #print('AT*z')
    #print(numpy.linalg.norm(numpy.dot(numpy.transpose(A), z)))
    #print('zusammen')
    #print(numpy.linalg.norm(a))
    b = numpy.subtract(numpy.dot(numpy.diag(s), z), my*numpy.ones(len(z)))
    #print('Sz-mye')
    #print(numpy.linalg.norm(b))
    c = numpy.linalg.norm(numpy.subtract(c_x, s))
    #print('C(x)-s')
    #print(c)
    text.write("%.4f %.4f %.4f" % (numpy.linalg.norm(a), numpy.linalg.norm(b), c))

    res = max(numpy.linalg.norm(a), numpy.linalg.norm(b), c)

    return res

def sqp_ipm(func, x, bdry, args, 
            maxiter=50,
            stepsize=1e-9,
            gradtol=1500,
            **kwargs):
    '''
    Does the SQP Algorithm and solves the quadratic problem with the quadratic
    interior point method (ipm_quadr)
    '''

    niter = 0
    n     = len(x)
    p     = len(bdry)

    #starting parameter
    y   = numpy.ones(p)
    lam = numpy.ones(p)
    while(1) :
        if (niter>=maxiter or numpy.linalg.norm(grad(func,x,stepsize))<=gradtol) :
            print('Stop of SQP Algorithm at Iteration=%d' % (niter))
            break

        niter = niter + 1
        print('ITERATON IN SQP =')
        print(niter)
        #set up matrices necessary for the ipm
        c = grad(func,x,stepsize)
        G = hessian(func,x)
        b = eval_c(x,bdry)
        A = numpy.zeros((p,n))
        for i in range(n) :
            A[2*i,i]   =  1
            A[2*i+1,i] = -1


        #Call ipm quadr
        print('HESSEMATRIX= ')
        print(G)
        print('EIGENWERTE DER HESSEMATRIX')
        w, v = numpy.linalg.eig(G)
        print(w)
        (x,y,lam) = ipm_quadr(G, c, A, b, x, y, lam)
    fk = func(x) 
    result = OptimizeResult(fun=fk, x=x, nit=niter)
    return result


def ipm_quadr(G, c, A, b, x, y, lam) :
    '''
    solves the optimization Problem of the form
    min f(x)= 0.5*x'*G*x + x'*c
    s.t. Ax >= b
    with the interior point method

    based on https://www.mcs.anl.gov/~anitescu/CLASSES/2012/LECTURES/S310-2012-lect11.pdf
    '''

    n = G.shape[1]  #len(x)
    p = len(b)      #amount constraints
    sigma = 0.5
    mu = numpy.dot(y,lam)/p
    def get_jac_quadr_ipm(x) :
        '''
        Computes the Jacobian for the  quadratic ipm
        based on https://www.mcs.anl.gov/~anitescu/CLASSES/2012/LECTURES/S310-2012-lect11.pdf
        '''
    
        Jac = numpy.zeros((n+p+p,n+p+p))

        Jac[0:n,0:n]     = G
        Jac[n:n+p,0:n]   = A

        Jac[n:n+p,n:n+p]     = -1.0*numpy.identity(p)
        Jac[n+p:n+p+p,n:n+p] = numpy.diag(x[n+p:n+p+p])

        Jac[0:n,n+p:n+p+p]      = -1.0*numpy.transpose(A)
        Jac[n+p:n+p+p,n+p:n+p+p] = numpy.diag(x[n:n+p])

        return Jac
    
    
    #set up the Function
    def nonlin_sys(x) :
        F = numpy.zeros(n+p+p)
        F[0:n]       = numpy.subtract(numpy.dot(G,x[0:n]), numpy.dot(numpy.transpose(A), x[n+p:n+p+p]))
        F[0:n]       = numpy.add(F[0:n], c)
        F[n:n+p]     = numpy.subtract(numpy.dot(A,x[0:n]), x[n:n+p])
        F[n:n+p]     = numpy.subtract(F[n:n+p], b)

        #Helper to calculate Y*LAM*e
        temp = numpy.zeros(p)
        for i in range(p) :
            F[i+n+p] = numpy.subtract(x[n+i]*x[n+p+i], sigma*mu)

        return numpy.transpose(F)

    #Setup x_ges = [x, y, lam]
    x_ges = numpy.zeros(n+p+p)
    x_ges[0:n]        = x
    x_ges[n:n+p]      = y
    x_ges[n+p:n+p+p]  = lam
    
    
    #print('JACOBIAN as array=')
    #test = numpy.asarray(get_jac_quadr_ipm(x_ges))
    #print(test)
    #nz_der = (test != 0)
    #print(nz_der)
    #print(test[nz_der])
    #fval = nonlin_sys(x_ges)
    #print(fval)
    #print(fval[nz_der])
    #Solve nonlinear system with newton
    #x_ges = optimize.newton(nonlin_sys, x_ges, maxiter=320)#, fprime=get_jac_quadr_ipm)
    x_ges  = newton_adj(nonlin_sys, get_jac_quadr_ipm, x_ges, 300, n, p)
    x   = x_ges[0:n]
    y   = x_ges[n:n+p]
    lam = x_ges[n+p:n+p+p]

    return (x,y,lam)

def newton_adj(F, J, x, maxiter, n, p, tol=1e-2) :
    '''
    newton mit daemofungsparameter, so dass y und lam positiv bleiben
    '''
    niter = 0
    while(1) :
        fval = F(x);
        if niter>=maxiter :
            print('newtonverfahren hat maximale Iteration errreicht, keine Konvergenz!')
            break
        elif numpy.linalg.norm(fval)<=tol :
            break
        niter = niter + 1
        #print('Niter in newton ist')
        #print(niter)
        df    = J(x)
        dfinv = inv(df)
        alpha = 1.0
        dx    = numpy.dot(dfinv,fval)
        
        while(1) : #for i in range(150) :
            xneu = x - alpha*dx
            print(xneu)
            if (any(n<0 for n in xneu[n:n+p+p])) :
                xneu = x    #go back to old x
                alpha = alpha/2
            else :
                print('alpha=')
                print(alpha)
                x = xneu
                break
    return x

    '''        
    for i in range(maxiter) :
        xneu = optimize.newton(F, x, maxiter=1)
        for j in range(30) :
            if (any(n<0 for n in x[n:n+p+p])) :
                dx = xneu-x
                xneu = x+(dx/2)
            else :
                break
        x = xneu
    return xneu    
    '''            


'''
#TEST
G = numpy.array([[10, 1], [1, 4]])
c = numpy.array([-4, -16])
A = numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b = numpy.array([-3, -6, -3, -6])

(x,y,lam) = ipm_quadr(G, c, A, b, [-1, 2], [1,1,1,1], [2,2,2,2]) 
for i in range(59) :
    print('x=')
    print(x)
    (x,y,lam) = ipm_quadr(G, c, A, b, x, y, lam)
    print('X')
    print(x)
'''

