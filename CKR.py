from __future__ import division
# -*- coding: utf-8 -*-

'''
Python implementation of CKR, an algorithm to compute globally accurate linear 
approximations for the solution of systems of nonlinear Ordinary Differential 
Equations (ODEs), under suitable conditions. 
It can be also used to compute overapproximations of reachsets for the 
considered system.

**Important: 
    - Python version 3.8.2 has been used. 
    - It requires MATLAB Engine API for Python.
    - The matlab script enc.m is required.
    
Some instructions:   
    - Both Python and Matlab files should be stored in the same directory.
    - The path of the directory has to be updated in Python file.
    - The analyzed non-linear system has to be specified in the matlab file enc.m
      before starting execution.
      
      
      
The main functions are:  
    
    - initvarAdvLong(Xlist,F,m,startmatlab): initializes global variables 
      and starts Matlab engine.
      
    - plotapprode(t1,x0init,tin, Xlist, j, col,step,npt=,thickness):
      computes and plots the approximate solution of the considered system.
        
    - AdvectPolytope(P,Xlist,F,m,T0,T1,krylov,specIntervals,epslist,numeric):
      overapproximates the reachset of the analyzed system,computing a list of polytopes 
      to approximate the reachtube for the considered time interval.
      


EXAMPLE OF CALLING SEQUENCE for reachsets computation, to be written at the end of the script:
    
    initialtimestep=.01
    timestep=initialtimestep 
    NL2=[4*y*(x + sqrt(3.0)), -4*(x + sqrt(3.0))**2 - 4*(y + 1)**2 + 16]
    initvarAdvLong([x,y],NL2,5,startmatlab=True)
    vp=[[0.45, 0.18], [0.45, 0.22], [0.52, 0.18], [0.52, 0.22]] 
    tl=AdvectPolytope(np.array(vp),[x,y],NL2,10,0,0.5,krylov=True,numeric=True,specIntervals=False)
    drawPipe2(tl,col='g')
            
Additional examples and experiments are at the end of the script.                            
    
'''

 


import scipy
import sympy
from sympy import *
from sympy.matrices import SparseMatrix
from scipy import sparse
from sympy.matrices.sparsetools import _doktocsr
import matplotlib
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.linalg import norm
from scipy import linalg as la
from scipy import linalg as LA
import pylab
import time
import random
import functools
from sympy import plot_implicit
import interval
from interval import interval, inf, imath
import mpmath
from mpl_toolkits.mplot3d import Axes3D
import sys
from random import random
from scipy.optimize import minimize
from z3 import *
from scipy.optimize import NonlinearConstraint 
import matplotlib.patches as patches
from fractions import Fraction as frac
import functools
import copy
init_printing(use_latex=False)



#variables
x, y, z, w, t , u, v, r= symbols('x y z w t u v r', real=True)
v1,v2, k, x1, x2, vx, vy= symbols('v1 v2 k x1 x2 vx vy', real=True)
a, b, c, d, e, f, g, h = symbols('a b c d e f g h')
a0, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 = symbols('a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14', real=True)
d1,d2,y1,y2,e1,e2,r1,r2,om,theta = symbols('d1,d2,y1,y2,e1,e2,r1,r2,om,theta', real=True)


# initial rectangle: 
x0low=-.05
x0high=.05
y0low=-.05
y0high=.05

#positivity threshold
eps=0.1
 

########################################################################


def grad(f, Xlist):
    '''
    It computes the gradient of a function.

    Args:
        - f: function.
        - Xlist: indipendent variables of the function.

    Returns:
        - g: gradient.

'''
    fe=Add(f,0)
    g=[]
    for x in Xlist:
        g.append(fe.diff(x))
    return(g)

  
def lie(p,F,Xlist):
    '''
    It computes Lie derivative of a function along a vector field.
    
    Args:
        - p: function.
        - F: list of expressions representing the considered vector field.
        - Xlist: list of indipendent variables.
    
    Returns:
        - the Lie derivative of p along F.

    '''
    i=0
    Nabla = grad(p,Xlist)
    e=0
    for d in Nabla:        
        e=e+d*F[i]
        i=i+1
    return sympy.simplify(e)
    

def polyvectorize(monlist,pe,Xlist):
    '''
    It converts a polynomial into vector form. 
    
    Args:
        monlist: list of monomials.
        pe:  polynomial expression.
        Xlist: list of independent variables.
    
    Returns:
        - v: polynomial expression in vector form.    
    '''
    p=Poly(pe,Xlist)
    poldic = p.as_dict()
    v = SparseMatrix(len(monlist),1,{})
    for term in poldic.keys():
            v[monlist.index(term),0]=poldic[term]
    return(v)



def trans(pelist,F,Xlist,m):
    '''
    It computes monomials and their coefficients reachable from a given expression
    doing Lie derivatives w.r.t a vector field for a fixed number of times.

    Args:
        - pelist: list of polynomial expressions.
        - F: list of expressions representing the considered vector field.
        - Xlist: list of independent variables.
        - m: integer representing the approximation order.

    Returns:
        - T: dictionary of monomials reachable from pelist doing Lie derivatives 
              w.r.t F for m times, and their Lie derivative.
    '''
    print('trans started...')
    T={}
    A=set({})
    for pe in pelist:
        p=Poly(pe,Xlist)
        A=A.union(set(p.monoms()))
    M=A
    for j in range(m-1):
        Mnew=set([])
        for mon in M:
            a=Poly({mon:1},Xlist)
            d=lie(a,F,Xlist)
            T[mon]=d.as_dict()
            mons=set(d.monoms())
            Mnew=Mnew.union(mons.difference(A))
        A=A.union(Mnew)
        M=Mnew
    for mon in M:
        a=Poly({mon:1},Xlist)
        d=lie(a,F,Xlist).as_dict()
        keys=set(d.keys())
        K = keys.intersection(A)
        T[mon]={ k:d[k] for k in K }
    print('trans finished.')
    return T

def matrixify(T):
    '''
    It splits the dictionary returned by trans into a matrix and a vector.
    
    Args:
        - T: dictionary returned by trans.
    Returns:
        - monlist: a list of monomials.
        - L: matrix whose columns contain coefficients of the Lie derivative 
             of monomials in monlist, truncated at m-1.

    '''
    monlist = list(T.keys())
    M = len(monlist)
    L = SparseMatrix(M,M,{})
    for j in range(M):
        poldic=T[monlist[j]]
        for term in poldic.keys():
            L[monlist.index(term),j]=poldic[term]
    return monlist,L   # L's column j=0,...,M-1 contains coeffs of lie derivative of j-th monomial in monlist (truncated for j=M-1)


def arnoldi(A,q1,m):
#   Arnoldi iteration
#   [Q,H] = ARNOLDI(A,q1,M) carries out M iterations of the
#   Arnoldi iteration with N-by-N matrix A and starting vector q1
#   (which need not have unit 2-norm).  For M < N it produces
#   an N-by-(M+1) matrix Q with orthonormal columns and an
#   (M+1)-by-M upper Hessenberg matrix H such that
#   A*Q(:,1:M) = Q(:,1:M)*H(1:M,1:M) + H(M+1,M)*Q(:,M+1)*E_M',
#   where E_M is the M'th column of the M-by-M identity matrix.

    N = A.shape[0]
    if (m>=N):
        return(eye(N),A)
    q1 = q1/q1.norm()
    Q = SparseMatrix(N,m+1,{}) 
    Q[:,0] = q1
    H = zeros(m+1,m+1)
    for k in range(1,m+1):
        z = A.multiply(Q[:,k-1])
        for i in range(1,k+1):
            H[i-1,k-1] = (Q[:,i-1].T).multiply(z)
            z = z - (Q[:,i-1]).scalar_multiply(H[i-1,k-1])
        if (k < N):
            H[k,k-1] = z.norm()
            if (H[k,k-1] == 0):
                print('lucky breakdown')
                return(Q.extract(range(0,Q.rows),range(0,k)),H.extract(range(0,k),range(0,k)))
            else:
                Q[:,k] = z/H[k,k-1]
    return(Q.extract(range(0,Q.rows),range(0,m)),H.extract(range(0,m),range(0,m)))


def numarnoldi(A,q1,m):
#   Arnoldi iteration
#   [Q,H] = ARNOLDI(A,q1,M) carries out M iterations of the
#   Arnoldi iteration with N-by-N matrix A and starting vector q1
#   (which need not have unit 2-norm).  For M < N it produces
#   an N-by-(M+1) matrix Q with orthonormal columns and an
#   (M+1)-by-M upper Hessenberg matrix H such that
#   A*Q(:,1:M) = Q(:,1:M)*H(1:M,1:M) + H(M+1,M)*Q(:,M+1)*E_M',
#   where E_M is the M'th column of the M-by-M identity matrix.

    N = A.shape[0]
    if (m>=N):
        return(eye(N),A)
    q1 = q1/scipy.linalg.norm(q1,ord=2)
    Q=sparse.dok_matrix((N,m+1),dtype=np.float64)
    Q[:,0] = q1
    H = np.zeros((m+1,m+1),dtype=np.float64)
    for k in range(1,m+1):
        z =  A*Q[:,k-1]
        for i in range(1,k+1):
            H[i-1,k-1] =  (Q[:,i-1].T)*z
            z = z -  Q[:,i-1]  * H[i-1,k-1]
        if (k < N):
            H[k,k-1] = LA.norm(z,ord=2)
            if (H[k,k-1] == 0):
                print('lucky breakdown')
                return Q[:,:k], H[:k,:k] 
            else:
                Q[:,k] = z/H[k,k-1]
    return Q[:,:m], H[:m,:m]  


###########################################


t=var('t')
tsm=type(SparseMatrix())

tol=10**(-7)
typeintv=type(interval([-1.0,1.0]))
delta = 0.01
def unpack(intlist):  
    'It flattens intervals representations (2D)'
    unpacked=[]
    for intv in intlist:
        if type(intv)==typeintv:
            unpacked.append(intv[0].inf)
            unpacked.append(intv[0].sup)
        else:
            unpacked.append(intv[0]) 
            unpacked.append(intv[1])
    return unpacked        




#################   Taylor expansions

t=symbols('t',positive=True)
def Taylor(g,F,Xlist,m):
    '''
    It computes the Taylor t-expansion of a function  of order m.
    
    Args:
    - g: function.
    - F: list of expressions representing the considered vector field.
    - Xlist: list of independent variables.
    - m: integer representing the approximation order.
    
    Returns:
        - the Taylor t-expansion of g.
    '''
    lielist=[g]
    f=g
    lj=g
    fact=1
    k=1
    for j in range(1,m):
        lj=lie(lj,F,Xlist)
        k=t*k/j
        f= (f+lj*k)
    return expand(f)

 


def TaylorAlt(g,Xlist,m):
    '''
    It computes the Taylor t-expansion of a function  of order m, alt matrix version.
    
    Args:
    - g: function.
    - Xlist: list of independent variables.
    - m: integer representing the approximation order.
    
    Returns:
        - the Taylor t-expansion of g, using global matrix L.
    '''
    global ml, L, x0
    v=polyvectorize(ml, g, Xlist)
    N=v.shape[0]
    Lj=SparseMatrix(N,m+1,{})
    Lj[:,0]=v
    ljv=v
    for j in range(1,m+1):
        ljv=L*ljv
        Lj[:,j]=ljv
    x0Lj=x0.T*Lj
    return sum([x0Lj[0,j]*t**j/factorial(j) for j in range(m)  ]), x0Lj[0,m]


 

 
############################ Symbolic Interval Arithmetic ###########################

constint=sympy.Function('constint')
sumint=sympy.Function('sumint')
prodint=sympy.Function('prodint')
powint=sympy.Function('powint')

 

def forceintv(r):
    if (type(r)!=list) and (type(r)!=tuple):
        return (r,r)
    return r

def fprodint(*ipars):
    ilist=list(ipars)
    if len(ilist)==0:
        return (1,1)
    if len(ilist)==1:
        return forceintv(ilist[0])
    if len(ilist)==2:
        i1=forceintv(ilist[0])
        i2=forceintv(ilist[1])
        return ( sympy.simplify(Min(i1[0]*i2[0],i1[0]*i2[1], i1[1]*i2[0],i1[1]*i2[1])), sympy.simplify(Max(i1[0]*i2[0],i1[0]*i2[1], i1[1]*i2[0], i1[1]*i2[1]))) 
    i2=fprodint(*ilist[1:])
    return fprodint(ilist[0],i2)

def fsumint(*ipars):
    ilist=list(ipars)
    if len(ilist)==0:
        return (0,0)
    if len(ilist)==1:
        return forceintv(ilist[0])
    if len(ilist)==2:
        i1=forceintv(ilist[0])
        i2=forceintv(ilist[1])
        return ( sympy.simplify(i1[0]+i2[0]),sympy.simplify(i1[1]+i2[1]) )
    i2=fsumint(*ilist[1:])
    return fsumint(ilist[0],i2)

def fpowint(i,k):
    if (type(k)==tuple) or (type(k)==list):
        k=k[0]
    if k<=1:
        return i
    return fprodint(*[i]*k)

siam={'sumint': fsumint, 'prodint': fprodint, 'powint': fpowint}


def predpow(e,symblist=[t]):
    if e.is_Pow:
        if not (e.base in symblist):
            return True
    return False

def translate(p,sl=[t]):
    p=expand(p)
    p=p.replace(Add,sumint)
    p=p.replace(Mul,prodint)
    pred=functools.partial(predpow,symblist=sl)
    p=p.replace(pred,lambda e: powint(e.base,e.exp))
    return p


def LIP(p,Xlist=[x,y],symblist=[t]):
    return lambdify(symblist+Xlist,translate(p,symblist),modules=[siam])

def ELIP(p,v, Xlist=[x,y],symblist=[t]):
    Fp= lambdify(symblist+Xlist,translate(p,symblist),modules=[siam])
    return Fp(*v)

def inflateint(intlist,rate=.1):
    inflatedintlist=[]
    for intv in intlist:
        l=intv[0]
        u=intv[1]       
        inflatedintlist.append([l-l*rate,u+u*rate])
    return inflatedintlist
 



vers=1
epsinflrate=.9
npoints=100
toval=500
def HOvalidateEnc2(Xlist,initint,enc,Delta,k,F=None,step=.01):
    '''
    It validates a given enclosure enc for flow, starting from the current Polytope.
    
    '''
    global xk, yk, xklead, yklead, fxklead, fyklead, ti, LX, UX, LY, UX
    if F!=None: # re-compute Taylor polynomial to order k
        xk=Taylor(x,F,Xlist,k+1)
        xklead=(Poly(xk,t).coeffs())[0]
        yk=Taylor(y,F,Xlist,k+1)
        yklead=(Poly(yk,t).coeffs())[0]
        fxklead=lambdify(Xlist,xklead)
        fyklead=lambdify(Xlist,yklead)                       
    LRx=fxklead(*enc)[0]
    LRy=fyklead(*enc)[0]
    
    pxI=fsumint(xk,fprodint([LRx[0],LRx[1]],t**(k+1)))
    lx=pxI[0]
    ux=pxI[1]
    
    pyI=fsumint(yk,fprodint([LRy[0],LRy[1]],t**(k+1)))
    ly=pyI[0]
    uy=pyI[1]
    
    z3v,z3e=sympy_to_z3([t,x,y], [lx,ux,ly,uy])
    z3t,z3x,z3y=z3v
    z3lx,z3ux,z3ly,z3uy=z3e
    x0min,x0max,y0min,y0max=unpack(initint)
    X0min,X0max,Y0min,Y0max=unpack(enc)
    cons=[z3x>=x0min,z3x<=x0max,z3y>=y0min,z3y<=y0max]
    
    T=Delta
    
    resFail=None
    modelFail=None
    condFail=None
    for cond,descr in zip([z3lx<=X0min,z3ux>=X0max,z3ly<=Y0min,z3uy>=Y0max],[0,1,2,3]):
        for time0 in np.arange(0,T,step):
            pr=Optimize()
            pr.add(cons)
            pr.add([z3t>=time0,z3t<=min(T,time0+step),cond])
            pr.set("timeout",toval)
            ob=pr.minimize(z3t)
            res=pr.check()
            if (res==sat) or (res==unknown):
                resFail=res
                modelFail=pr.model()
                condFail=descr
                if time0==0:
                    return False,(descr,res), modelFail
                T=time0
                break
    return T, (condFail,resFail), modelFail


inceps=.01
def iterateHOvalidateEnc2(Xlist,initint,tin,Delta,k,epslist,F=None,step=.01):
    '''
    It computes a validated enclosure for flow, starting from the current Polytope without matlab engine.
    
    Args:
        - Xlist: list of independent variables.
        - initint: bounding box for current vertices.
        - tin: initial time.
        - Delta: length of the timestep.
        - k: degree of the Taylor polynomial.
        - epslist: list of bounds for absolute value of Lagrange remainder,
          one for each coordinate.
                 
          
    Returns:
        - time, enc, a validated enclosure and the corresponding time.
   '''
    global xk, yk, xklead, yklead, fxklead, fyklead, ti, LX, UX, LY, UX

    idx0=find_nearest(ti,tin)
    idx1=find_nearest(ti,tin+Delta)
    A,B,C,D=(min(LX[idx0:idx1+1]),max(UX[idx0:idx1+1]),min(LY[idx0:idx1+1]),max(UY[idx0:idx1+1]))
    
    for j in range(ntries):
        enc=[interval([A-epslist[0],B+epslist[1]]), interval([C-epslist[2],D+epslist[3]])]
        time, res, model = HOvalidateEnc2(Xlist,initint,enc,Delta,k,F=F,step=.01)
        if time==Delta:
            return time, enc
        j=res[0]
        epslist[j]=epslist[j]+inceps
    return time, enc    

    


import matlab.engine
slack=10**-4
 

        
def updateAdvfunc(Xlist,m,krylov=False):
    '''
    It updates the advection function after each step.
    
    Args:
        - Xlist: list of independent variables.
        - m: order of approximation.
        - krylov: boolean variable, if true, the dimension reduction of 
          the system is done via Krylov projection.
    '''
    global timestep,Advlist,Ftlist,LagrangeFactor
    Delta=timestep
    LagrangeFactor=Delta**(m+1)/factorial(m) 
    for i in range(len(Ftlist)):
        if krylov:  
            fxi=Ftlist[i](Delta)
        else:
            fxi=Ftlist[i](Delta) 
        Advlist[i]=lambdify(Xlist,fxi)   
    return None
            

def Adv(v):
    '''
    It computes the advection function of a given vector.
        Args:
            - v: the vector to advect.
        Returns:
            - the advection function computed in v.
    '''
    global Advlist
    return [ fadv(*v) for fadv in Advlist]
        
        
def LagrangeRem(enc,m,Delta):       
    '''
    It computes bounds for the Lagrange remainder in absolute value.
        Args:
            - enc: validated enclosure.
            - m: order of approximation.
            - Delta: considered time.
        Returns:
            - epslist: list of bounds for absolute value of Lagrange remainder,
              one for each coordinate.
    '''  
    global Rlist,LagrangeFactor
    epslist=[]
    LF=(Delta**(m+1))/factorial(m+1)
    for Ri in Rlist:
        LR=Ri(*enc)[0]
        epslist=epslist+[max(np.abs(LR.inf),np.abs(LR.sup))*LF]
    return epslist

'''
def LagrangeRemList(enc,m,Delta):      
    global Rlist,LagrangeFactor
    epslist=[]
    LF=(Delta**(m+1))/factorial(m+1)
    for Ri in Rlist:
        LR=Ri(*enc)[0]
        epslist.append((LR.inf*LF,LR.sup**LF))
    return epslist
'''

import polytope as pc
def validatedEnclosure(currentVertices,Delta,intv=False): 
    '''
    It computes a validated enclosure for flow, starting from the current Polytope and invoking matlab engine.
    
    Args:
        - currentVertices: array representing the initial rectangle.
        - Delta: length of the considered time interval.
          
    Returns:
        - hyper-rectangle representing a validated enclosure
   '''
    global errml,eng
    if intv:
        intervals=currentVertices
    else:
        intervals=boundingBox(currentVertices)
    bb=matlab.double([list(r) for r in np.array(intervals)])
    res=eng.enc(bb,Delta,nargout=1,stderr=errml)
    return [interval(list(c)) for c in np.array(res)]

nvertices=0

uppernvertices=0
def AdvectPolytope(P,Xlist,F,m,T0,T1,krylov=False,specIntervals=False,epslist=[.05]*4,numeric=True):
    '''
     It computes a list of polytopes to approximate the reachtube for the considered time interval.
       
     Args:
             - P: array representing the initial rectangle.
             - Xlist: list of independent variables.
             - F: list of expressions representing the considered vector field.
             - m: integer representing the approximation order.
             - T0: left end of the considered time interval.
             - T1: right end of the considered time interval. 
             - krylov: boolean variable, if true, the dimension reduction of the system is done via Krylov projection.
             - specIntervals: boolean variable, used to convert the representation of the initial rectangle,
               from inf-sup representation to list of intervals representation.
             - epslist: list of bounds for absolute value of Lagrange remainder,
               one for each coordinate.
                 
           
     Returns:
             - reachtubelist: list of polytopes and vertices to build the reachtube from time T0 to T1.
                 
    '''  
    global uppernvertices,T,ml,L,Lnum,x0, initialtimestep, timestep, ti, Rlist,Vlist,Hlist,Advlist,eng, nvertices, uppernvertices
    start_time = time.time()  
    timestep=initialtimestep
    updateAdvfunc(Xlist,m,krylov)
    tleft=T0
    if specIntervals:
        P=hyperRect(P,Xlist,2,offset=50)
    elif specIntervals=='infsup':
        P=hyperRect(P,Xlist,2,False,offset=50)
    currentPolytope=pc.qhull(np.array(P))
    currentVertices=pc.extreme(currentPolytope)
    nvertices=len(currentVertices)
    uppernvertices=2*nvertices
    reachtubelist=[(currentPolytope,0,currentPolytope,currentVertices,currentVertices,None)]
    if methencl==1:
        initint=boundingBox(currentVertices)
        epsl=epslist.copy()
    
    while (T1-(tleft+timestep))>=-tol:
        if timestep<tol:
            print('Timestep under tolerance, premature termination.')
            print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
            return reachtubelist
        if (methencl==1) and (norm(epsl,inf)>norm(accuracy,inf)):
            print('Considered enclosing box too large, premature termination.')
            print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
            return reachtubelist
        print('')
        print('*Solving piece* from ',tleft,' to ',tleft+timestep)
        print('    Computing validated enclosure...')
        
        Delta=timestep
        try:
            if methencl==1:
                Delta, enc =iterateHOvalidateEnc2(Xlist,initint,tleft,timestep,1,epslist=epsl,step=.01)
            else:
                enc=validatedEnclosure(currentVertices,timestep)

        except Exception: # as excp:
            if timestep<tol: 
                print('Cannot find enclosure, premature termination.')
                print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
                return reachtubelist
            timestep=timestep/2
            updateAdvfunc(Xlist,m,krylov)
            print('    Cannot find enclosure, @halving timestep@.')
            continue
        if (Delta==False) or (Delta==0):
            if methencl==1:
                print('    Cannot find enclosure, considering larger box....')
                epsl=[2*e for e in epsl]             
                continue
            else:
                print('Cannot find enclosure, premature termination.')
                print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
                return reachtubelist
        print('    Enclosure found: ',enc)
        print('    Actual current timestep =',Delta)        
        remBoundList=LagrangeRem(enc,m,Delta)   
        advectedVertices=np.array([Adv(v) for v in currentVertices]) 

        try:            
            res=inflatePolytope(currentPolytope,currentVertices,advectedVertices,remBoundList)#inflatePolytope(advectedVertices,epslist)  # builds convex hull of advected vertices, then inflates it taking into account epslist (Lagr. remainder) AND convexity errors, then returns list of vertices of inflated polytope           
        except Exception:
            if timestep<tol: 
                print('Cannot find polytope, premature termination.')
                print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
                return reachtubelist
            timestep=timestep/2
            updateAdvfunc(Xlist,m)
            print('    Cannot find polytope, @halving timestep@.')
            continue             
        
        if type(res)==bool:
            print('Polytope may be degenerating, premature termination.')
            print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
            return reachtubelist
        
        newPolytope,newVertices=res        
        reachtubelist.append((currentPolytope,Delta,newPolytope,currentVertices,newVertices,enc.copy()))                
        currentPolytope=newPolytope
        currentVertices=newVertices
        if methencl==1:
            initint=boundingBox(currentVertices)
        tleft=tleft+Delta
    print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
    return reachtubelist


from scipy.optimize import  differential_evolution
from scipy.optimize import LinearConstraint, Bounds

def boundingBox(Vertices):
    '''
    It builds a bounding box, computing a list of intervals, one for each coordinate.
    
    Args:
        - Vertices: vertices of the current convex polytope.
    Returns:
        - bb: list of intervals, one for each coordinate.
    
    '''
    bb=[]
    for j in range(Vertices.shape[1]):
        bb.append( (min(Vertices[:,j]), max(Vertices[:,j]) ) ) 
    return bb   

def hyperRect(intervalList,Xlist,k,offset=0,infsup=False,outputInt=False): 
    '''
    It computes the vertices of the initial polytope.
    
    Args:
        - intervalList: list of intervals, one for each coordinate.
        - Xlist: list of independent variables
        - k: integer, s.t. 10^k is a scaling factor, large enough to make all numbers in interval list integers.
    
    Returns:
        - cfsn: the list of vertices of the initial polytope.
    '''
    if infsup:
        intervalList=[list(r) for r in np.array(intervalList).T]   
        if outputInt:
            return intervalList
    for x,e in zip(Xlist,intervalList):
        print(e[1],e[0])

    p=prod([x**(int(offset+e[0]*10**k))+x**(int(offset+e[1]*10**k)) for x,e in zip(Xlist,intervalList)]) 
    cfs=list(Poly(p,Xlist).as_dict().keys())
    cfsn=[[(a-offset)/10**k for a in e] for e in cfs]
    print(cfsn)
    return cfsn
  

def polytopeReduction(p,meth='girard'):  
    '''
    It reduces the number of vertices of a given polytope.
    
    Args:
        - p: the polytope to reduce.
    
    Returns:
        - the reduced polytope.
    '''
    global nvertices, Advlist
    n=len(Advlist)
    cb=np.empty(shape=(p.b.shape[0],1))
    cb[:,0]=p.b
    A=matlab.double([list(r) for r in p.A])
    b=matlab.double([list(r) for r in cb])
    res=np.array(eng.reducePT(A,b,nvertices/n,meth,nargout=1,stderr=errml))
    redA=res[:,:-1]
    redb=res[:,-1]
    return pc.Polytope(redA,redb)
  
multoffset=1.05
degeneracycheck=True
def inflatePolytope(currentPolytope,currentVertices,advectedVertices,epslist=[],tol=10**-5): 
    '''
    It inflates the current polytope, maximizing the advection of the vertices of the current polytope
    along direction of the unit vector c.
    
    Args: 
        - currentPolytope: current polytope in H-representation.
        - currentVertices: vertices of the current polytope.
        - advectedVertices: advection function computed in currentVertices.
    
    Returns:
        - newPolytope: inflated polytope.
        - newVertices: inflated vertices.
    '''
    global Advlist, nvertices, uppernvertices, degeneracycheck, multoffset
    
    A=currentPolytope.A
    b=currentPolytope.b
    lc=LinearConstraint(A, -np.inf,b,  keep_feasible=False)
    bounds=boundingBox(currentVertices)#currentPolytope bounding box
    newP=pc.reduce(pc.qhull(advectedVertices))
    nrepairs=0
    
    while True:
        if nrepairs>=5:
            return False
        vnP=pc.extreme(newP)
        C=newP.A#.copy()
        d=newP.b#.copy()

        if degeneracycheck & (len(vnP)<nvertices):   # take a recovery action if n. of verteces decreas
            nrepairs+=1
            for v,k in zip(advectedVertices,range(len(advectedVertices))):  # move faces outward to include v
                if not (v in vnP):   
                    print('found advected vector not in c.h.',v)      # must push faces outward to include v
                    distvector=np.array([C[i,:].dot(v)-d[i] for i in range(C.shape[0])]) # search for hyperplane of minimal distance from v
                    i=np.argmin(distvector)
                    di=distvector[i]
                    c=C[i,:]      
                    v=v+multoffset*di*c 
                    advectedVertices[k]=v 
                    newP=pc.reduce(pc.qhull(advectedVertices))
                    break
            continue  # if n. of vertices was found decreased, skip the remaining part of the while iteration and restarts
            
        # here n. of vertices of newP  == nvertices
        for i in range(C.shape[0]):  # for each face
            c=C[i,:]      # c is *outward* pointing normal vector
            f=lambda v: -np.array(Adv(v)).dot(c)
            print('optimization for face ',i,'...')
            res = differential_evolution(f, bounds,  constraints=(lc),tol=tol) # maximal magintude of Adv(v) along c direction for v in currentPolytope
            print('finished. ')
            projectederr=np.abs(np.array(epslist).dot(c))
            print('projected Lagrange rem. error',projectederr)
            d[i]=-res.fun+projectederr # update vector of distances d
        newPolytope=pc.reduce(pc.Polytope(C, d))
        newVertices=pc.extreme(newPolytope)
        if newVertices is None:
            print('no new vertices.')    
            return False
        if degeneracycheck & (len(newVertices)<nvertices):
            print('n. of vertices descreased, trying to recover by rotating two faces...')
            nrepairs+=1
            newP=pc.reduce(rotateFace(newP)) # restarts cycle with rotated faces
            continue
        if degeneracycheck & (len(newVertices)>uppernvertices):
            print('n. of vertices',len(newVertices),' exceeding predefined threshold, applying reduction...')
            newPolytope=polytopeReduction(newP) # invoke CORA's reduction method
            newVertices=pc.extreme(newPolytope)
            if newVertices>uppernvertices:
                print('n. of vertices',len(newVertices),' still exceeding predefined threshold')
                return False # fail to reduce
        print('n. of new vertices',len(newVertices))
        return newPolytope,newVertices




def rotate(v,i,j,theta): 
    'It rotates v of angle theta counterclockwise on plane xi-xj'
    global Advlist
    n=len(Advlist)
    R=np.eye(n)
    R[i,i]=np.cos(theta)
    R[i,j]=-np.sin(theta)
    R[j,i]=np.sin(theta)
    R[j,j]=np.cos(theta)
    return R.dot(v)

thetaoffset=(2*np.pi)/30
def rotateFace(p):   
    'It slightly rotates on opposite directions two faces whose hyperplane is nearly the same, on the plane x-y.'
    b=p.b.copy()
    A=p.A.copy()
    vp=pc.extreme(p)
    maxsp=-np.inf   # look for a pair of rows of maximal inner product (nearly parallel)
    for i in range(A.shape[0]):
        for k in range(i+1,A.shape[0]):
            dik=A[i].dot(A[k])
            if maxsp<dik:
                maxik=(i,k)
                maxsp=dik
    i,k=maxik
    c1=A[i].copy()
    c2=A[k].copy()
    r1=rotate(c1,0,1,thetaoffset) # rotate counter-clockwise
    if r1.dot(c2)>maxsp:
        r1=rotate(c1,0,1,-thetaoffset) # rotate clockwise if necessary
        r2=rotate(c2,0,1,thetaoffset)
    else:
        r2=rotate(c2,0,1,-thetaoffset)
    A[i]=r1  # now normal of faces i and k is rotated conunterclockwise of thetaoffset on x-y plane
    A[k]=r2
    for r,l in zip([r1,r2],[i,k]):
        for v in vp: #check that all old vertices are included in the new rotated halfplanes, if not move halfplane outward by adding an offset   
            d=r.dot(v)-b[l]
            if d>0:
                print(v,'outside of ',d)
                b[l]=b[l]+d   
    return pc.Polytope(A,b)     

slackfig=.1
def plotP(p,col='b',lw=1):
    'It plots a single polytope.'
    points=pc.extreme(p)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=col,linewidth=lw)

from scipy.spatial import ConvexHull

    
def drawPipe2(tl,time=None,col='b',lw=2,dim=None):
    'It plots the sequence of polytopes that builds the reachtube'
    if time==None:
        polytopeList=[e[2] for e in tl]#+[tl[-1][2]]
    else:
        polytopeList=[]
        currentTime=0
        for e in tl:
            if currentTime>=time:
                break
            polytopeList.append(e[2])
            currentTime+=e[1]
    if dim!=None:
        polytopeList=[q.project(dim) for q in polytopeList]
    plt.figure()
    for pv in polytopeList:
        plotP(pv,col=col,lw=lw)
    #plt.show()


import warnings
#warnings.filterwarnings("error")
from sklearn.decomposition import PCA,FastICA    

def find_nearest(array, value):
    'It subtracts value from each element of the array and takes the minimum'
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx #array[idx]



#################### LONG ADVECTION #################################
maxdegree=12
ZZ=var('@#Z')
monlist=[m for m in itermonomials([x,y],maxdegree)]
lenmonlist=len(monlist)
levelset=[10**-5]
stepgrid=.01

gridx=0.01
gridy=0.01

#ntries=2
accuracy=[2]*4
methencl=2

def computeh(Xlist,F):
    '''
    It computes h functions for error bounding.
    Args:
        - Xlist: list of independent variables.
        - F: list of expressions representing the considered vector field.
    
    Returns:
        - hlist: list h functions.    
    '''
    global Lnum,x0,vlist,Vlist
    hlist=[]
    for v,V in zip(vlist,Vlist):
        vm=V[:,-1]
        ld=lie((x0.T*vm)[0],F,Xlist)
        w=L.multiply(vm)
        pr=(w.T).multiply(V).multiply(V.T).multiply(x0)[0]
        h=ld-pr
        hlist.append(h)
    return hlist   

import scipy.integrate as integrate
def initvarAdvLong(Xlist,F,m,startmatlab=False):
    '''
    It initializes global variables and starts Matlab engine.      
        Args:
            - Xlist: list of independent variables.
            - F: list of expressions representing the considered vector field.
            - m: integer representing the approximation order.
            - startmatlab: boolean variable, if true, Matlab is used to produce
              a validated enclosure, otherwise not.
        ''' 
    global T,ml,L,Lnum,x0, timestep, stpcontx, stpconty,Rlist,vlist,Ftlist,Fxlist,vlist,Vlist,Hlist,Advlist,eng,errml,LagrangeFactor,VF,hlist,fhlist,integratorlist
    timestep=initialtimestep
    Delta=timestep
    stpcontx=gridx
    stpconty=gridy
    print('Start building basis of monomials (coordinates of tensor space)')
    T=trans(Xlist,F,Xlist,m) # T = transition relation on all N monomials reachable from g within m-1 steps
    ml,L = matrixify(T) # L = transition relation in matrix form; column j=0,1,...,M-1 contains coeffs. of next-state of j-th monomial in ml (truncated, for the j=M-1). NB: corresponds to L in (Boreale,Fossacs'17) and to A^T in (Boreale,HSCC'18) 
    x0=SparseMatrix(len(ml),1,{})
    x0[:,0]=Matrix([Poly({m:1},Xlist)/1 for m in ml]) # x0=column vector of all monomials   
    Lnum=np.array(L).astype(np.float64)
    
    # for bounding the Lagrange remainder
    Rlist=[]
    Advlist=[]
    LagrangeFactor=Delta**(m+1)/factorial(m)
    k=len(Xlist)
    W,W1,W2,Vsp,Vlist,vlist,vnlist,Hlist,HT,expHT,fxilist,Ftlist=[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k,[0]*k
    for xi,i in zip(Xlist,range(k)):
        _,ri=TaylorAlt(xi,Xlist,m+2)
        Ri=lambdify(Xlist,ri)
        Rlist=Rlist+[Ri]
        print('Compute linear advection operator for ', xi)
        print('Build representation of ', xi, ' in tensor space' )
        vlist[i]=polyvectorize(ml,xi,Xlist) # v = column vector of coefficients of g in the basis ml    
        print('Invoke Arnoldi to build matrices H,V for ', xi)
        vnlist[i]=np.array(vlist[i]).astype(np.float64)        
        Vlist[i],Hlist[i]=numarnoldi(Lnum,vnlist[i],m) 
        m=Hlist[i].shape[0]        
        Hlist[i]=np.matrix(Hlist[i]).astype(np.float64) 
        HT[i]=Hlist[i].T
        expHT[i]=expm(HT[i]*Delta) #  matrix exp
        Vsp[i]=SparseMatrix(*Vlist[i].shape,dict(Vlist[i])).copy()
        W1[i]=(vlist[i].T).multiply(Vsp[i])
        W2[i]=W1[i].multiply(expHT[i])  
        W[i]=W2[i].multiply(Vsp[i].T)
        fxilist[i]=W[i].multiply(x0)[0,0]
        Ftlist[i]=lambda t,i=i: W1[i].multiply(expm(HT[i]*t)).multiply(Vsp[i].T).multiply(x0)[0,0]#Ftlist+[fxt]
        advi=lambdify(Xlist,fxilist[i])
        Advlist=Advlist+[advi]
        
        
    print('Compute h functions for error bounding')
    hlist=computeh(Xlist,F)
    fhlist=[lambdify(Xlist,h) for h in hlist]
    
    print('Compute integrator functions: lambda t0,t1: int_t0^t1 |exp((t1-tau)*H.T)[1,m]| dtau, for each H')
    integratorlist=[ lambda t0,t1, t:  integrate.quad(lambda tau: np.abs(scipy.linalg.expm((t-tau)*H.T)[0,-1]), t0, t1)[0] for H in Hlist]
    
    if startmatlab:
        print('Start Matlab engine')
        eng = matlab.engine.start_matlab()
        eng.cd(r'C:\Users\Utente\Dropbox\topics\SuperReach\ReachTubes', nargout=0)
        eng.ls(nargout=0)
        eng.initmatlab(nargout=0)
        errml = io.StringIO()

############### Utility: convert sympy expr to Z3 expr


from z3 import Real, Sqrt 
from sympy.core import Mul, Expr, Add, Pow, Symbol, Number

def sympy_to_z3(sympy_var_list, sympy_exp_list):
    'convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)'

    z3_vars = []
    z3_var_map = {}

    for var in sympy_var_list:
        name = 'z3'+var.name
        z3_var = Real(name)
        z3_var_map[name] = z3_var
        z3_vars.append(z3_var)
    
    resultlist=[ _sympy_to_z3_rec(z3_var_map, sympy_exp) for sympy_exp in sympy_exp_list]

    return z3_vars, resultlist

def _sympy_to_z3_rec(var_map, e):
    'recursive call for sympy_to_z3()'

    rv = None

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, Symbol):
        rv = var_map.get('z3'+e.name)

        if rv == None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, Number):
        rv = float(e)
    elif isinstance(e, Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])

        if exponent == 0.5:
            # sqrt
            rv = Sqrt(term)
        else:
            rv = term**exponent

    if rv == None:
        raise RuntimeError("Type '" + str(type(e)) + "' is not yet implemented for convertion to a z3 expresion. " + \
                            "Subexpression was '" + str(e) + "'.")

    return rv  


##############################  Plot functions  ############################
              
def plotode(F, t1, tin=0, Xlist=[x,y], j=-1, col='b', x0init=None,step=.01,npt=npoints,thickness=1.0):
    '''
    It plots the exact solution of the considered system, computed numerically.
    
    Args:
    - F: list of expressions representing the considered vector field.
    - t1: right end of the considered time interval.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - j: coordinate of the function to be plotted, if j=-1 the x-y space
         is considered for the plotting.
    - col: string representing the colour of the plotted curve.
    - x0init: initial point for the solution.
    '''
    tt = np.linspace(tin, t1, npt)
    
    fF=[lambdify(Xlist,f) for f in F]
    vf= lambda u, t: np.array([f(*u) for f in fF])
    
    if x0init!=None:
        sol = odeint(vf, x0init, tt)
        if j!=-1:
            plt.plot(tt,sol[:,j],color=col,linewidth=thickness)
        else:
            plt.plot(sol[:,0],sol[:,1],color=col,linewidth=thickness)
        return True
 


def plotapprode_2ind(t1,  x0init,tin=0, Xlist=[x,y], j=-1, col='b',step=.01,npt=npoints,thickness=1.0):
    '''
    It plots our approximate solution considering two independent variables.
    
    Args:
    - t1: right end of the considered time interval.
    - x0init: initial point for the approximate solution.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - j: coordinate of the function to be plotted, if j=-1 the x-y space
         is considered for the plotting.
    - col: string representing the colour of the plotted curve.
    '''
    
    
    global x0, Vlist, Hlist
    
    #a,b=x0init
    xx0=x0.subs({x:a for x,a in zip(Xlist,x0init)})
    y00T=(xx0.T).multiply(Vlist[0])
    y00=y00T.T
    f0=lambda t: (scipy.linalg.expm(t*Hlist[0].T)*y00)[0]
    y01T=(xx0.T).multiply(Vlist[1])
    y01=y01T.T
    f1=lambda t: (scipy.linalg.expm(t*Hlist[1].T)*y01)[0]
    
    timeint = np.linspace(tin, t1, npt)
    solx=[f0(tt) for tt in timeint]
    soly=[f1(tt) for tt in timeint]
    sols=[solx,soly]
    if j!=-1:
        plt.plot(timeint,sols[j],color=col,linewidth=thickness)
    else:
        plt.plot(solx,soly,color=col,linewidth=thickness)
        

def plotapprode(t1,  x0init,tin=0, Xlist=[x,y,z,w,u,v,r], j=-1, col='b',step=.01,npt=npoints,thickness=1.0):
    '''
    It plots our approximate solution considering seven independent variables.
    
    Args:
    - t1: right end of the considered time interval.
    - x0init: initial point for the approximate solution.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - j: coordinate of the function to be plotted, if j=-1 the x-y space
         is considered for the plotting.
    - col: string representing the colour of the plotted curve.
    '''
    global x0, Vlist, Hlist  
    
    #a,b=x0init
    xx0=x0.subs({x:a for x,a in zip(Xlist,x0init)})
    
    y00T=(xx0.T).multiply(Vlist[0])
    y00=y00T.T
    f0=lambda t: (scipy.linalg.expm(t*Hlist[0].T)*y00)[0]
    
    y01T=(xx0.T).multiply(Vlist[1])
    y01=y01T.T
    f1=lambda t: (scipy.linalg.expm(t*Hlist[1].T)*y01)[0]
    
    y02T=(xx0.T).multiply(Vlist[2])
    y02=y02T.T
    f2=lambda t: (scipy.linalg.expm(t*Hlist[2].T)*y02)[0]
    
    y03T=(xx0.T).multiply(Vlist[3])
    y03=y03T.T
    f3=lambda t: (scipy.linalg.expm(t*Hlist[3].T)*y03)[0]
    
    y04T=(xx0.T).multiply(Vlist[4])
    y04=y04T.T
    f4=lambda t: (scipy.linalg.expm(t*Hlist[4].T)*y04)[0]
    
    y05T=(xx0.T).multiply(Vlist[5])
    y05=y05T.T
    f5=lambda t: (scipy.linalg.expm(t*Hlist[5].T)*y05)[0]
    
    y06T=(xx0.T).multiply(Vlist[6])
    y06=y06T.T
    f6=lambda t: (scipy.linalg.expm(t*Hlist[6].T)*y06)[0]
    
    timeint = np.linspace(tin, t1, npt)
    #x, y, z, w, u, v, r
    solx=[f0(tt) for tt in timeint]
    soly=[f1(tt) for tt in timeint]
    solz=[f2(tt) for tt in timeint]
    solw=[f3(tt) for tt in timeint]
    solu=[f4(tt) for tt in timeint]
    solv=[f5(tt) for tt in timeint]
    solr=[f6(tt) for tt in timeint]
    
    sols=[solx,soly,solz,solw,solu,solv,solr]
    if j!=-1:
        plt.plot(timeint,sols[j],color=col,linewidth=thickness)
    else:
        plt.plot(solx,soly,color=col,linewidth=thickness)
    

def plottaylorode_2ind(t1,  x0init,m,tin=0, Xlist=[x,y], j=-1, col='b',step=.01,npt=npoints,thickness=1.0):
    '''
    It plots the Taylor expansion of a fixed order of the solution, considering two independent variables.
        
    Args:
    - t1: right end of the considered time interval.
    - x0init: initial point for the approximate solution.
    - m: integer representing the order of the Taylor expansion.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - j: coordinate of the function to be plotted, if j=-1 the x-y space
         is considered for the plotting.
    - col: string representing the colour of the plotted curve.
    '''
    global x0, Vlist, Hlist
    
    ftx,_=TaylorAlt(Xlist[0],Xlist,m)
    fty,_=TaylorAlt(Xlist[1],Xlist,m)
    a,b=x0init
    fx=ftx.subs({Xlist[0]:a, Xlist[1]:b})
    fy=fty.subs({Xlist[0]:a, Xlist[1]:b})
    f0=lambdify([t],fx)
    f1=lambdify([t],fy)
    
    
    timeint = np.linspace(tin, t1, npt)
    solx=[f0(tt) for tt in timeint]
    soly=[f1(tt) for tt in timeint]
    sols=[solx,soly]
    if j!=-1:
        plt.plot(timeint,sols[j],color=col,linewidth=thickness)
    else:
        plt.plot(solx,soly,color=col,linewidth=thickness)    
    

def plottaylorode(t1,  x0init,m,tin=0, Xlist=[x,y,z,w,u,v,r], j=-1, col='b',step=.01,npt=npoints,thickness=1.0):
    '''
    It plots the Taylor expansion of a fixed order of the solution, considering seven independent variables.
        
    Args:
    - t1: right end of the considered time interval.
    - x0init: initial point for the approximate solution.
    - m: integer representing the order of the Taylor expansion.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - j: coordinate of the function to be plotted, if j=-1 the x-y space
         is considered for the plotting.
    - col: string representing the colour of the plotted curve.
    '''

    global x0, Vlist, Hlist
    ftx,_=TaylorAlt(Xlist[0],Xlist,m)
    fty,_=TaylorAlt(Xlist[1],Xlist,m)
    ftz,_=TaylorAlt(Xlist[2],Xlist,m)
    ftw,_=TaylorAlt(Xlist[3],Xlist,m)
    ftu,_=TaylorAlt(Xlist[4],Xlist,m)
    ftv,_=TaylorAlt(Xlist[5],Xlist,m)
    ftr,_=TaylorAlt(Xlist[6],Xlist,m)
    a,b,c,d,e,f,g=x0init
    fx=ftx.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fy=fty.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fz=ftz.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fw=ftw.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fu=ftu.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fv=ftv.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})
    fr=ftr.subs({Xlist[0]:a, Xlist[1]:b, Xlist[2]:c, Xlist[3]:d, Xlist[4]:e, Xlist[5]:f,Xlist[6]:g})


    f0=lambdify([t],fx)
    f1=lambdify([t],fy)
    f2=lambdify([t],fz)
    f3=lambdify([t],fw)
    f4=lambdify([t],fu)
    f5=lambdify([t],fv)
    f6=lambdify([t],fr)
    
    
    timeint = np.linspace(tin, t1, npt)
    solx=[f0(tt) for tt in timeint]
    soly=[f1(tt) for tt in timeint]
    solz=[f2(tt) for tt in timeint]
    solw=[f3(tt) for tt in timeint]
    solu=[f4(tt) for tt in timeint]
    solv=[f5(tt) for tt in timeint]
    solr=[f6(tt) for tt in timeint]
    
    sols=[solx,soly,solz,solw,solu,solv,solr]
    if j!=-1:
        plt.plot(timeint,sols[j],color=col,linewidth=thickness)
    else:
        plt.plot(solx,soly,color=col,linewidth=thickness)    


def plotlinearizedOde(t1, F, x0init,tin=0, Xlist=[x,y], mod=-1, i=0,j=1, col='b',step=.01,npt=npoints,thickness=1.0):    
    '''
    It plots the the solution of the linearized system.
        
    Args:
    - t1: right end of the considered time interval.
    - F: list of expressions representing the considered vector field.
    - x0init: initial point for the approximate solution.
    - tin: left end of the considered time interval. 
    - Xlist: list of independent variables.
    - mod: integer that determines the plot method, if mod==1 the xi-xj plane 
           is considered for the plotting, othewise the t-xi plane is considered.
    - i: coordinate of the function to be plotted.
    - j: coordinate of the function to be plotted.
    - col: string representing the colour of the plotted curve.
    '''
    J=Matrix([grad(f,Xlist) for f in F])
    sigma0={x:x0i for x,x0i in zip(Xlist,x0init)}
    J0=np.array(J.subs(sigma0),dtype=np.float64)
    lF=[lambdify(Xlist,v) for v in np.array([f.subs(sigma0) for f in F])+J0.dot([x-x0i for x,x0i in zip(Xlist,x0init)])]
    Flin=lambda u,t :np.array([v(*u) for v in lF])
    tt = np.linspace(tin, t1, npt)
    sol=odeint(Flin,x0init,tt)
    if mod!=-1:
        plt.plot(tt,sol[:,i],color=col,linewidth=thickness)
    else:
        plt.plot(sol[:,i],sol[:,j],color=col,linewidth=thickness)


def plotlinearizedOde(t1, F, x0init,tin=0, Xlist=[x,y], mod=-1, i=0,j=1, col='b',step=.01,npt=npoints,thickness=1.0):    
    J=Matrix([grad(f,Xlist) for f in F])
    sigma0={x:x0i for x,x0i in zip(Xlist,x0init)}
    J0=np.array(J.subs(sigma0),dtype=np.float64)
    lF=[lambdify(Xlist,v) for v in np.array([f.subs(sigma0) for f in F])+J0.dot([x-x0i for x,x0i in zip(Xlist,x0init)])]
    Flin=lambda u,t :np.array([v(*u) for v in lF])
    tt = np.linspace(tin, t1, npt)
    sol=odeint(Flin,x0init,tt)
    if mod!=-1:
        plt.plot(tt,sol[:,i],color=col,linewidth=thickness)
    else:
        plt.plot(sol[:,i],sol[:,j],color=col,linewidth=thickness)


 
 
    
 
#-----------------------------------------------------------------------------------------   
#-----------------------------------------------------------------------------------------   
#---------------------------------- Experiments ------------------------------------------   
#-----------------------------------------------------------------------------------------   
#-----------------------------------------------------------------------------------------   




#-----------------------------------------------------------------------------------------   
#-------------------------------- Graphical comparisons ----------------------------------
#-----------------------------------------------------------------------------------------   


#------------------------------- 28(a) ---------------------------------------
'''
initialtimestep=.01
timestep=initialtimestep 

NL2=[4*y*(x + sqrt(3.0)), -4*(x + sqrt(3.0))**2 - 4*(y + 1)**2 + 16]
initvarAdvLong([x,y],NL2,10,startmatlab=True)
vp=[[0.45, 0.18], [0.45, 0.22], [0.52, 0.18], [0.52, 0.22]] 
tl=AdvectPolytope(np.array(vp),[x,y],NL2,10,0,1,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')  


import random

drawPipe2(tl,col='g')   
for i in range(20):
    x, y = random.uniform(4.5/10,5.2/10),random.uniform(1.8/10,2.2/10)
    plotode(NL2,1,x0init=[x,y],col='gold',thickness=0.2)   # fig. di dx
plt.show() 

plotlinearizedOde(1,NL2,x0init=[4.8/10,2/10], col='g',thickness=3) 
plotode(NL2,1,x0init=[4.8/10,2/10],col='gold',thickness=3)   # fig. di dx
plotapprode_2ind(1,[4.8/10,2/10],thickness=3,col='k')
plottaylorode_2ind(.2,[4.8/10,2/10],10,thickness=3,col='b')
plt.show()
'''


#------------------------------- 28(b) ---------------------------------------
'''
initialtimestep=.05
timestep=initialtimestep 

LV=[x*(1.5 - y), -y*(3 - x)] 
initvarAdvLong([x,y],LV,5,startmatlab=True)
vp=[[0.45, 0.18], [0.45, 0.22], [0.52, 0.18], [0.52, 0.22]] 
tl=AdvectPolytope(np.array(vp),[x,y],LV,5,0,1,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g') 


import random

for i in range(20):
    x, y = random.uniform(4.5/10,5.2/10),random.uniform(1.8/10,2.2/10)
    plotode(LV,1,x0init=[x,y],col='gold',thickness=0.2)    
plt.show()
 
x0small=[0.45, 0.22]
plotlinearizedOde(1,LV,x0init=[4.5/10, 2.2/10], col='limegreen',thickness=3) 
plotapprode_2ind(1,x0small,thickness=3,col='k')
plotode(LV,1,x0init=x0small,col='y',thickness=3)
plottaylorode_2ind(1,x0small,5,thickness=3,col='b')
plt.show()
'''



#--------------------------------------- LL -------------------------------------------
'''
initialtimestep=.001
timestep=initialtimestep
LL=[1.4*z-0.9*x, 2.5*u-1.5*y, 0.6*r-0.8*y*z, 2-1.3*w*z, 0.7*x-w*u, 0.3*x-3.1*v, 1.8*v-1.5*r*y ]


#------X_1
initvarAdvLong([x,y, z, w, u, v, r],LL,5,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=0,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=0,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=0, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=0, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_1(t)$')
plt.xlabel('t')
plt.show()



#------X_2
initvarAdvLong([x,y, z, w, u, v, r],LL,4,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=1,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=1,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=1, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=1, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_2(t)$')
plt.xlabel('t')
plt.show()


#------X_3
initvarAdvLong([x,y, z, w, u, v, r],LL,5,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=2,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=2,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=2, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=2, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_3(t)$')
plt.xlabel('t')
plt.show()


#------X_4
initvarAdvLong([x,y, z, w, u, v, r],LL,5,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=3,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=3,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=3, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=3, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_4(t)$')
plt.xlabel('t')
plt.show()


#------X_5
initvarAdvLong([x,y, z, w, u, v, r],LL,4,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=4,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=4,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=4, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=4, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_5(t)$')
plt.xlabel('t')
plt.show()


#------X_6
initvarAdvLong([x,y, z, w, u, v, r],LL,5,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=5,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=5,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=5, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=5, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_6(t)$')
plt.xlabel('t')
plt.show()



#------X_6
initvarAdvLong([x,y, z, w, u, v, r],LL,5,startmatlab=True)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=6,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=6,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=6, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=6, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_7(t)$')
plt.xlabel('t')
plt.show()

'''


#-----------------------------------------------------------------------------------------  
#---------------------------------- Reachsets: table 1 ------------------------------------ 
#-----------------------------------------------------------------------------------------  


#---------------------------------- 28(b) -------------------------------------
#--------- th=1
'''
initialtimestep=.05
timestep=initialtimestep

LV=[x*(1.5 - y), -y*(3 - x)] 
initvarAdvLong([x,y],LV,4,startmatlab=True)
vp=[[0.40, 0.18], [0.40, 0.27], [0.52, 0.18], [0.52, 0.27]]
tl=AdvectPolytope(np.array(vp),[x,y],LV,4,0,1,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2,16])
plt.ylim([-4,12])
plt.savefig('C:/Users/Utente/Desktop/img.svg',dpi=350)
plt.show()
'''


#--------- t=3
'''
initialtimestep=.05
timestep=initialtimestep

LV=[x*(1.5 - y), -y*(3 - x)] 
initvarAdvLong([x,y],LV,4,startmatlab=True)
vp=[[0.40, 0.18], [0.40, 0.27], [0.52, 0.18], [0.52, 0.27]]
tl=AdvectPolytope(np.array(vp),[x,y],LV,4,0,3,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2,16])
plt.ylim([-4,12])
plt.savefig('C:/Users/Utente/Desktop/img.svg',dpi=350)
plt.show()
'''


#--------- th=5
'''
initialtimestep=.05
timestep=initialtimestep

LV=[x*(1.5 - y), -y*(3 - x)] 
initvarAdvLong([x,y],LV,5,startmatlab=True)
vp=[[0.40, 0.18], [0.40, 0.27], [0.52, 0.18], [0.52, 0.27]]
tl=AdvectPolytope(np.array(vp),[x,y],LV,5,0,5,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2,16])
plt.ylim([-4,12])
plt.savefig('C:/Users/Utente/Desktop/img.svg',dpi=350)
plt.show()
'''

#------------------------------------- (30) ------------------------------------------------

#--------- th=1
'''
initialtimestep=.05
timestep=initialtimestep

STB =[-x**3+y,-x**3-y**3]
initvarAdvLong([x,y],STB,5,startmatlab=True)
vp=[[-0.5, -0.7], [-0.5, 0.8], [0.3, -0.7], [0.3, 0.8]] 
tl=AdvectPolytope(np.array(vp),[x,y],STB,5,0,1,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.show()
'''


#--------- th=3
'''
initialtimestep=.05
timestep=initialtimestep

STB =[-x**3+y,-x**3-y**3]
initvarAdvLong([x,y],STB,4,startmatlab=True)
vp=[[-0.5, -0.7], [-0.5, 0.8], [0.3, -0.7], [0.3, 0.8]] 
tl=AdvectPolytope(np.array(vp),[x,y],STB,4,0,3,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.show()
'''


#--------- th=5
'''
initialtimestep=.05
timestep=initialtimestep

STB =[-x**3+y,-x**3-y**3]
initvarAdvLong([x,y],STB,4,startmatlab=True)
vp=[[-0.5, -0.7], [-0.5, 0.8], [0.3, -0.7], [0.3, 0.8]] 
tl=AdvectPolytope(np.array(vp),[x,y],STB,4,0,5,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.show()
'''



#-------------------------------------- VDP ---------------------------------------------
#---------  th=1
'''
initialtimestep=.05
timestep=initialtimestep
VDP = [y,(1-x**2)*y-x]
initvarAdvLong([x,y],VDP,4,startmatlab=True)
vp=[[1,2],[1.5,2],[1.5,2.45],[1,2.45]]

tl=AdvectPolytope(np.array(vp),[x,y],VDP,5,0,1,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2.5,2.5])
plt.ylim([-3,5])#3

plt.xticks(np.arange(-2.5, 2.5+0.5, 0.5))
plt.yticks(np.arange(-3, 5+1, 1))

plt.ylabel('y')
plt.xlabel('x')
plt.show()
'''


#---------  th=3
'''
initialtimestep=.05
timestep=initialtimestep
VDP = [y,(1-x**2)*y-x]
initvarAdvLong([x,y],VDP,4,startmatlab=True)
vp=[[1,2],[1.5,2],[1.5,2.45],[1,2.45]] 

tl=AdvectPolytope(np.array(vp),[x,y],VDP,4,0,3,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2.5,2.5])
plt.ylim([-3,5])#3

plt.xticks(np.arange(-2.5, 2.5+0.5, 0.5))
plt.yticks(np.arange(-3, 5+1, 1))

plt.ylabel('y')
plt.xlabel('x')
plt.show()
'''


#---------  th=5
'''
initialtimestep=.05
timestep=initialtimestep
VDP = [y,(1-x**2)*y-x]
initvarAdvLong([x,y],VDP,5,startmatlab=True)
vp=[[1,2],[1.5,2],[1.5,2.45],[1,2.45]] #vertices of initial set, a rectangle

tl=AdvectPolytope(np.array(vp),[x,y],VDP,5,0,5,krylov=True,numeric=True,specIntervals=False)
drawPipe2(tl,col='g')   
plt.xlim([-2.5,2.5])
plt.ylim([-3,5])#3

plt.xticks(np.arange(-2.5, 2.5+0.5, 0.5))
plt.yticks(np.arange(-3, 5+1, 1))

plt.ylabel('y')
plt.xlabel('x')
plt.show()
'''
