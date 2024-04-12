'''
Python implementation of CKR, an algorithm to compute linear approximations 
for the solution of systems of nonlinear Ordinary Differential  Equations (ODEs).
It can be also used to compute overapproximations of reachsets for the 
considered system.

**Important: 
    - Python version 3.8.2 has been used for the experiments, but it is not
      mandatory to replicate them.
    - File conversions.py of pypolycontain library has been modified adding the
      control option: 'qhull_options="QJ"' at the ConvexHull call at line 162,
      obtaining: ConvexHull(v, qhull_options="QJ").
      
      
The main functions are:  
    
    - initvarAdvLong(Xlist,F,m,startmatlab): initializes global variables 
      and compute the approximation of the system and the corresponding error.

        
    - iterReach(approx,[x,y],[Z0.x,Z0.G],200,4,checkden=True):
      overapproximates the reachset of the analyzed system,computing a list of zonotopes
      to approximate the reachtube for the considered time interval.
      


EXAMPLE OF CALLING SEQUENCE for reachsets computation, to be written at the end of the script:
    
    
    initialtimestep=.05
    timestep=initialtimestep
    cBRU = [1+x**2*y-1.5*x-x , 1.5*x-x**2*y]
    initvarAdvLong([x,y],cBRU,4)
    TL=TaylorList([x,y],cBRU,5) 
    Z0=pp.zonotope(x=np.array([0.9,0.1]).reshape(2,1), G=np.array([[0.1,0],[0,0.1]]).reshape(2,2))
    RS=iterReach(approx,[x,y],[Z0.x,Z0.G],200,4,checkden=True)

            
Additional examples and experiments are at the end of the script.                            
    
'''
import numpy as np
from mpmath import iv    # for Interval Arithmetic
import pypolycontain as pp  # for zonotopes
import matplotlib
from sympy import *
import copy
import time
import scipy
import sympy
from sympy import *
from sympy.matrices import SparseMatrix
from scipy import sparse
from sympy.matrices.sparsetools import _doktocsr
import matplotlib
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pylab
import time
import random
import functools
import mpmath
from mpl_toolkits.mplot3d import Axes3D
import sys
from random import random
from z3 import *
import pypolycontain as pp  # for zonotopes
from scipy.optimize import  differential_evolution
from sympy.utilities.autowrap import ufuncify
from scipy.integrate import quad  
from scipy.optimize import LinearConstraint, Bounds
from scipy.optimize import linprog   
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.linalg import norm
from scipy import linalg as LA
from scipy.optimize import minimize


#variables
x, y, z, w, t , u, v, r= symbols('x y z w t u v r', real=True)

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
    


npoints=100

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
    return monlist,L   

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




slack=10**-4
maxdegree=12
ZZ=var('@#Z')
monlist=[m for m in itermonomials([x,y],maxdegree)]
lenmonlist=len(monlist)
levelset=[10**-5]
stepgrid=.01
gridx=0.01
gridy=0.01
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



def initvarAdvLong(Xlist,F,m):
    '''
    It initializes global variables and starts Matlab engine.      
        Args:
            - Xlist: list of dependent variables.
            - F: list of expressions representing the considered vector field.
            - m: integer representing the approximation order.
        ''' 
    global T,ml,L,Lnum,x0, timestep, stpcontx, stpconty,Rlist,vlist,Ftlist,Fxlist,vlist,Vlist,Hlist,Advlist,eng,errml,LagrangeFactor,VF,hlist,fhlist,integratorlist, approx
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
    #integratorlist=[ lambda t0,t1, t:  quad(lambda tau: np.abs(scipy.linalg.expm((t-tau)*H.T)[0,-1]), t0, t1)[0] for H in Hlist]
    integratorlist=[ lambda t0,t1, t:  quad(lambda tau: (scipy.linalg.expm((t-tau)*H.T)[0,-1]), t0, t1)[0] for H in Hlist]
    approx = fxilist



def rotate(v,i,j,theta): 
    '''
    It rotates a vector v of an angle theta counterclockwise on plane xi-xj and returns the rotated vector.
    '''
    n=v.shape[0]
    R=np.eye(n)
    R[i,i]=np.cos(theta)
    R[i,j]=-np.sin(theta)
    R[j,i]=np.sin(theta)
    R[j,j]=np.cos(theta)
    return R.dot(v)




def iterReach(flist,Xlist,Z0,N,m,tol=10**-2,checkden=False,eps=.1,theta=1.3,Moore=False):
    '''
     It computes a list of zonotopes to approximate the reachtube for the considered time interval.
       
     Args:
             - flist: list of expressions representing the considered vector field.
             - Xlist: list of independent variables.
             - Z0: array representing the initial zonotope.
             - N: integer representing the number of iterations. 
             - m: integer representing the approximation order.
           
     Returns:
             - RS: list of zonotopes representing the reachtube.
                 
    '''
    global timestep, integratorlist,TL
    start_time=time.time()
    c=timestep**m/factorial(m)
    print('Lagrange factor Delta^m/m!=',c)
    n=len(Xlist)
    
    advL=[lambdify(Xlist,fe) for fe in flist]
    adv = lambda v: np.array([f(*v) for f in advL])
    intFactors=[integratorlist[i](0,timestep,timestep) for i in range(n)]
        
    reachsetList=[Z0]
    C=Z0
    k=C[1].shape[1]
    lambdastar=[]
    for j in range(N):
        advGenplus=adv((C[0]+C[1]))
        advGenminus=adv((C[0]-C[1]))
        centroid = ((advGenplus+advGenminus).sum(axis=1)/(2*k)).reshape(n,1)
        try:
            newGen=advGenplus-centroid
        except:
            print("Problems with generators and center")
            return reachsetList
        
        if checkden:
            try:
                nGN=newGen/norm(newGen,axis=0)
            except:
                print("Problems with generators")
                return reachsetList
            Gram=nGN.T.dot(nGN)  # Gramian matrix of nGN: Gram[i,j] == cos(thetaij), where thetaij is angle between gi and gj 
            for i in range(n):
                for p in range(i):
                    if Gram[i,p]>=1-eps:   # gi and gj parallel, same direction
                        newGen[:,p]=rotate(newGen[:,p],i,p,theta).reshape(n,) # rotate gi counterclockwise of theta
                    if Gram[i,p]<=-1+eps:  # gi and gj parallel, opposite directions
                        newGen[:,p]=rotate(newGen[:,p],i,p,-theta).reshape(n,) # rotate gi clockwise of theta

        C1=[centroid, newGen]
        try:
            Gpi=scipy.linalg.pinv(C1[1],atol=10**-2)
        except:
            print(C1)
            print('Error')
            return reachsetList
        lambdastar=[]
        for i in range(k):
            fvi=lambda v,i=i,Gpi=Gpi,C=C,C1=C1: Gpi[i,:]@( adv((C[0]+C[1]@(v.reshape(k,1))))-C1[0])
            res = scipy.optimize.shgo(fvi, ((-1,1),)*k) 
            if res.success==False:
                print('optimization failed')
                print(res)
                return reachsetList
            lambdastar.append(-res.fun)
        print('lambdastar=',lambdastar)

        Z0=pp.zonotope(x=C[0],G=C[1])
        B=bounding_box(Z0)
        Aiv=npToIntv(B)
        Biv=iter_mooreValid(TL,Aiv,Aiv,timestep,k=10)
      
        delta=Encl(Gpi,n,Biv,intFactors)
        print('delta=',delta)
        for i in range(len(lambdastar)):
           if not isinstance( lambdastar[i], float):
               lambdastar[i]=(lambdastar[i])[0] 
               
        L=np.eye(k)*(np.array(lambdastar)+np.array(delta))
        C1inf=[C1[0],C1[1]@L]
        reachsetList.append(C1inf)
        C=C1inf
    RS=[pp.zonotope(x=Z[0],G=Z[1]) for Z in reachsetList]
    try:            
        fig,ax=plt.subplots()
        plt.xlim([-0.1, 0.1]) 
        fig.set_size_inches(6, 3)
        pp.visualize(RS,fig,ax)
        ax.set_title(r'A triangle (red), rotated by 30 degrees (blue), and a zonotope (green)')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.axis('equal')    
    except:
        ()
    print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
    return RS




def splitZonotope(Z,i): 
    '''
    It splits a zonotope Z along generator gi and returns the two splits.
    '''
    Z1=copy.deepcopy(Z)
    Z2=copy.deepcopy(Z)
    m=Z.G.shape[0]
    gi=Z.G[:,i].reshape(m,1)
    Z1.x=Z1.x+gi/2
    Z1.G[:,i]=(gi/2).reshape(m,)
    Z2.x=Z2.x-gi/2
    Z2.G[:,i]=(gi/2).reshape(m,)
    return Z1,Z2



def bounding_box(Z):
    '''
    It returns a bounding box of zonotope Z.
    '''
    v=pp.zonotope_to_V(Z)
    min_value= np.min(v, axis=0)
    max_value = np.max(v, axis=0) 
    bounding_box=[]
    bounding_box = np.array([np.array([min_value, max_value]) for min_value, max_value in zip(min_value, max_value)])
    return bounding_box

     

def Encl(Gpi,n,B,intFactors):
    '''
    It computes an enclosure of the local error and returns delta (inflation).
    '''
    global   fhlist 
    intvlist=[iF*fh(*B) for iF,fh in zip(intFactors,fhlist)]
    bounds=  [(float(intv.a),float(intv.b))  for intv in intvlist] 
    print('bounds=',bounds)
    delta=  ([np.abs(linprog(-Gpi[i,:], bounds=bounds).fun) for i in range(Gpi.shape[0])])
    #delta= np.array([float(np.abs(fun( *(list(-Gpi[i,:])+bounds))).b) for i in range(Gpi.shape[0])])
    return delta



def drawReach(RS,border=.96,alpha=0.8,dims=[0,1]):
    L=[]
    for Z in RS:
        Z.color='blue'
        L.append(Z)
        Z0=copy.deepcopy(Z)
        Z0.color='green'
        k=Z0.G.shape[1]
        Z0.G=Z.G@(np.eye(k)*border)
        L.append(Z0)
    try:
        pp.visualize(L,title='Reachsets',alpha=alpha,tuple_of_projection_dimensions=dims)     # visualize C1inf
    except:
        ()
        

def npToIntv(A):
    return [iv.mpf(r) for r in A]


#############################  MOORE ALGORITHM FOR ENCLOSURE COMPUTATION  ##############################

def TaylorList(Xlist,F,m):
    pdList=[Poly(Taylor(x,F,Xlist,m+1),t).as_dict() for x in Xlist]
    LRlst= [pd[(m,)] for pd in pdList]
    for pd in pdList:
        del(pd[(m,)])
    TPolyList=[Poly(pd,t)/1 for pd in pdList]
    
    X1list=list(var(' '.join([str(x)+'1' for x in Xlist] )) )
    sigma={x:x1 for x,x1 in zip(Xlist,X1list)}
    ifact=1/factorial(m)
    TLf = [ lambdify([t]+Xlist+X1list,TPolyList[i] + ifact*LRlst[i].subs(sigma)*t**m ) for i in range(len(Xlist))  ]
    return TLf
    
def checkIncl(I,J):
    return (float(I.a)>=float(J.a)) & (float(I.b)<=float(J.b))
    
def mooreValid(TaylorList,Aiv,Biv,Delta,alpha=.5,tol=0.001):
    if Delta<tol:
        print("Fail")
        return None,None
    Div=iv.mpf([0,Delta])
    newIntv=[T(*([Div]+Aiv+Biv)) for T in TaylorList]
    checkList=[checkIncl(I,J) for I,J in zip(newIntv,Biv)]
    if all(checkList):
        print('Delta=',Delta)
        return [T(*([Div]+Aiv+newIntv)) for T in TaylorList],Delta
    Biv=newIntv
    newIntv=[T(*([Div]+Aiv+newIntv)) for T in TaylorList]
    checkList=[checkIncl(I,J) for I,J in zip(newIntv,Biv)]
    if all(checkList):
        print('Delta=',Delta)
        return [T(*([Div]+Aiv+newIntv)) for T in TaylorList], Delta
    return mooreValid(TaylorList,Aiv,Biv,Delta*alpha,alpha,tol)
    

def iter_mooreValid(TaylorList,Aiv,Biv,Delta,i=0,k=2):
    if i>=k:
        print("Fail")
        return None
    Div=iv.mpf([0,Delta])
    newIntv=[T(*([Div]+Aiv+Biv)) for T in TaylorList]
    checkList=[checkIncl(I,J) for I,J in zip(newIntv,Biv)]
    if all(checkList):
        print('   Moore iter=',i+1)
        return [T(*([Div]+Aiv+newIntv)) for T in TaylorList]
    #Biv=newIntv
    return iter_mooreValid(TaylorList,Aiv,newIntv,Delta,i+1,k)


def mooreCheck(TL,Aiv,Biv,Delta):   
    Div=iv.mpf([0,Delta])
    newIntv=[T(*([Div]+Aiv+Biv)) for T in TL]
    checkList=[checkIncl(I,J) for I,J in zip(newIntv,Biv)]
    return all(checkList)
    
#------------------------------------------------------



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
        
def plotapprode_vdp(t1,  x0init,tin=0, Xlist=[x,y], j=-1, col='b',step=.01,npt=npoints,thickness=1.0):
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
#-------------------------------------- Experiments -------------------------------------- 
#-----------------------------------------------------------------------------------------  



#--------------------------------- VDP---------------------------------------
'''
initialtimestep=.001
timestep=initialtimestep
vdp= [(1-x**2)*y-x,y]
initvarAdvLong([x,y],vdp,2)
plotapprode_vdp(2,[0.1,0.1],j=0,Xlist=[x,y],thickness=2,col='k')
plt.ylabel(r'$\tilde{x}_1(t)$')
plt.xlabel('t')
plt.show()
'''


#------------------------------- 28(a) ---------------------------------------
'''
initialtimestep=.01
timestep=initialtimestep 

NL2=[4*y*(x + sqrt(3.0)), -4*(x + sqrt(3.0))**2 - 4*(y + 1)**2 + 16]
initvarAdvLong([x,y],NL2,10)

plotlinearizedOde(1,NL2,x0init=[4.8/10,2/10], col='g',thickness=3) 
plotode(NL2,1,x0init=[4.8/10,2/10],col='gold',thickness=3)   # fig. di dx
plotapprode_2ind(1,[4.8/10,2/10],thickness=3,col='k')
plottaylorode_2ind(.2,[4.8/10,2/10],10,thickness=3,col='b')
plt.show()
'''


#------------------------------- 28(b) ----------------------------------------
'''
initialtimestep=.05
timestep=initialtimestep 

LV=[x*(1.5 - y), -y*(3 - x)] 
initvarAdvLong([x,y],LV,5)

x0small=[0.45, 0.22]
plotlinearizedOde(1,LV,x0init=[4.5/10, 2.2/10], col='limegreen',thickness=3) 
plotapprode_2ind(1,x0small,thickness=3,col='k')
plotode(LV,1,x0init=x0small,col='y',thickness=3)
plottaylorode_2ind(1,x0small,5,thickness=3,col='b')
plt.show()

'''



#--------------------------------------- LL -----------------------------------
'''
initialtimestep=.001
timestep=initialtimestep
LL=[1.4*z-0.9*x, 2.5*u-1.5*y, 0.6*r-0.8*y*z, 2-1.3*w*z, 0.7*x-w*u, 0.3*x-3.1*v, 1.8*v-1.5*r*y ]


#------X_1
initvarAdvLong([x,y, z, w, u, v, r],LL,5)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=0,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=0,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=0, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=0, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_1(t)$')
plt.xlabel('t')
plt.show()




#------X_2
initvarAdvLong([x,y, z, w, u, v, r],LL,4)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=1,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=1,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=1, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=1, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_2(t)$')
plt.xlabel('t')
plt.show()


#------X_3
initvarAdvLong([x,y, z, w, u, v, r],LL,5)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=2,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=2,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=2, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=2, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_3(t)$')
plt.xlabel('t')
plt.show()


#------X_4
initvarAdvLong([x,y, z, w, u, v, r],LL,5)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=3,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=3,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=3, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=3, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_4(t)$')
plt.xlabel('t')
plt.show()


#------X_5
initvarAdvLong([x,y, z, w, u, v, r],LL,4)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=4,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=4,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=4, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=4, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_5(t)$')
plt.xlabel('t')
plt.show()


#------X_6
initvarAdvLong([x,y, z, w, u, v, r],LL,5)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=5,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=5,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=5, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=5, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_6(t)$')
plt.xlabel('t')
plt.show()



#------X_7
initvarAdvLong([x,y, z, w, u, v, r],LL,5)
plottaylorode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],5,Xlist=[x,y, z, w, u, v, r],j=6,thickness=2,col='b')
plotapprode(1,[0.4,0.1,0.2,1.0,0.2,0.4,0.1],j=6,Xlist=[x,y, z, w, u, v, r],thickness=2,col='k')
plotlinearizedOde(1,LL,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r], mod=0, i=6, col='g',thickness=2) 
plotode(LL,1,x0init=[0.4,0.1,0.2,1.0,0.2,0.4,0.1],Xlist=[x,y, z, w, u, v, r],j=6, col='y',thickness=2) 
plt.ylabel(r'$\tilde{x}_7(t)$')
plt.xlabel('t')
plt.show()
'''





#-----------------------------------------------------------------------------------------  
#---------------------------------- Reachsets: table 2 ------------------------------------ 
#-----------------------------------------------------------------------------------------  



#-------------------------------------JET
'''
initialtimestep=.05
timestep=initialtimestep
cJET = [-y-1.5*x**2-0.5*x**3-0.5, 3*x-y]    
initvarAdvLong([x,y],cJET,5)

TL=TaylorList([x,y],cJET,5) 

Z0=pp.zonotope(x=np.array([1,1]).reshape(2,1), G=np.array([[0.2,0],[0,0.2]]).reshape(2,2))

f_cjet=[f(timestep) for f in Ftlist]

start_time=time.time()
RS=iterReach(f_cjet,[x,y],[Z0.x,Z0.G],200,5,checkden= False)
final_time=(time.time()-start_time)
print('ex. time:',final_time)
sum([Z.volume() for Z in RS])/200
RS[200].volume()
'''

#----------------------------------BRU
'''
initialtimestep=.05
timestep=initialtimestep
cBRU = [1+x**2*y-1.5*x-x , 1.5*x-x**2*y]
initvarAdvLong([x,y],cBRU,4)

TL=TaylorList([x,y],cBRU,5) 


Z0=pp.zonotope(x=np.array([0.9,0.1]).reshape(2,1), G=np.array([[0.1,0],[0,0.1]]).reshape(2,2))
f_bru = approx
start_time=time.time()
RS=iterReach(f_bru,[x,y],[Z0.x,Z0.G],200,4,checkden=True)

final_time=(time.time()-start_time)
print(final_time)
sum([Z.volume() for Z in RS])/200
RS[200].volume()

'''


#-------------------------------------Lorenz 
'''
var('x y z w')


initialtimestep=.01
timestep=initialtimestep
lor = [10*(y-x), x*(8/3-z)-y, x*y-28*z]
initvarAdvLong([x,y,z],lor,4)
Z0=pp.zonotope(x=np.array([15,15,35]).reshape(3,1), G=np.array([[0.1,0,0],[0,0.1,0],[0,0,0.1]]).reshape(3,3)) 

TL=TaylorList([x,y,z],lor,5) 

f_lor = approx
Zh1,Zh2=splitZonotope(Z0,0)
Zhlu,Zh2ld=splitZonotope(Zh1,1)
Zhru,Zh2rd=splitZonotope(Zh2,1)
start_time=time.time()
RS_lu=iterReach(f_lor,[x,y, z],[Z0.x,Z0.G],200,4,checkden=False)
final_time=(time.time()-start_time)


drawReach(RS_lu)
print(final_time)
sum([Z.volume() for Z in RS_lu])/200
RS_lu[200].volume()
'''
##---------------------------------cVDP

'''
initialtimestep=.05
timestep=initialtimestep
cVDP = [y,(1-x**2)*y-x+(w-x),z,(1-w**2)*z-w+(x-w)]
initvarAdvLong([x,y,w,z],cVDP,4)

TL=TaylorList([x,y,w,z],cVDP,5) 


Z0=pp.zonotope(x=np.array([0,0.500,0,0.500]).reshape(4,1), G=np.array([[0.025,0,0,0],[0,0.025,0,0],[0,0,0.025,0],[0,0,0,0.025]]).reshape(4,4))

f_cvdp=[f(timestep) for f in Ftlist]

start_time=time.time()
RS=iterReach(f_cvdp,[x,y,w,z],[Z0.x,Z0.G],120,4,checkden=False)
final_time=(time.time()-start_time)
print(final_time)

sum([Z.volume() for Z in RS])/120
RS[120].volume()
'''


#----------------------------------------------LV
'''
initialtimestep=0.08

timestep=initialtimestep
cLV = [x*(1-(x+0.85*y+0.5*u)), y*(1-y+(0.85*z+0.5*x)),z*(1-(z+0.85*w+0.5*y)),w*(1-(w+0.85*u+0.5*z)) ,u*(1-(u+0.85*x+0.5*w))]
initvarAdvLong([x,y, z, w, u],cLV,3)

TL=TaylorList([x,y, z, w, u],cLV,5) 

Z0=pp.zonotope(x=np.array([1,1,1,1,1]).reshape(5,1), G=np.array([[0.05,0,0,0,0],[0,0.05,0,0,0],[0,0,0.05,0,0],[0,0,0,0.05,0],[0,0,0,0,0.05]]).reshape(5,5)) 
f_lv =[f(timestep) for f in Ftlist]
Zh1,Zh2=splitZonotope(Z0,0)

timestep=.08
f_lv=[f(timestep) for f in Ftlist]
start_time=time.time()
RS1=iterReach(f_lv,[x,y, z, w, u],[Zh1.x,Zh1.G],63,3,checkden=True)
RS2=iterReach(f_lv,[x,y, z, w, u],[Zh2.x,Zh2.G],63,3,checkden=True)
final_time=(time.time()-start_time)


sum([Z.volume() for Z in RS1])/63+sum([Z.volume() for Z in RS2])/63
RS1[63].volume()+RS2[63].volume()
'''

#----------------------------------VDP
'''
Z0=pp.zonotope(x=np.array([1.25,2.225]).reshape(2,1), G=np.array([[0.25,0],[0,0.225]]).reshape(2,2))
Zh1,Zh2=splitZonotope(Z0,0)
Zhlu,Zh2ld=splitZonotope(Zh1,1)
Zhru,Zh2rd=splitZonotope(Zh2,1)


initialtimestep=.02
timestep=initialtimestep
cVDP = [y,(1-x**2)*y-x]    
initvarAdvLong([x,y],cVDP,4)


TL=TaylorList([x,y],cVDP,5) 
R1=[]
R2=[]


f_vdp =[f(timestep) for f in Ftlist]

vol_tot=0
Zh1,Zh2=splitZonotope(Z0,0)


f_cvdp = approx
start_time=time.time()
RS1=iterReach(f_vdp,[x,y],[Zh1.x,Zh1.G],350,4,checkden= False)
RS2=iterReach(f_vdp,[x,y],[Zh2.x,Zh2.G],350,4,checkden= False)

final_time=(time.time()-start_time)
print('ex. time:',final_time)

sum([Z.volume() for Z in RS1])/350 +sum([Z.volume() for Z in RS2])/350

RS1[350].volume()+RS2[350].volume()
'''
#---------------------------------------ROSS
'''
initialtimestep=.02
timestep=initialtimestep
ROS = [-y-z,x+0.2*y,0.2+z*(x-5.7)]
Z0=pp.zonotope(x=np.array([0,-8.40,0]).reshape(3,1), G=np.array([[0.2,0,0],[0,0.2,0],[0,0,0.2]]).reshape(3,3)) 
Zh1,Zh2=splitZonotope(Z0,0)
Zhlu,Zh2ld=splitZonotope(Zh1,1)
Zhru,Zh2rd=splitZonotope(Zh2,1)
initvarAdvLong([x,y, z],ROS,4)
TL=TaylorList([x,y, z],ROS,5) 

timestep=0.02

f_ros=[f(timestep) for f in Ftlist]

start_time=time.time()
RS_lu=iterReach(f_ros,[x,y, z],[Zh1.x,Zh1.G],300,4,checkden=False)
RS_ld=iterReach(f_ros,[x,y, z],[Zh2.x,Zh2.G],300,4,checkden=False)
final_time=(time.time()-start_time)

print(final_time)

v=sum([Z.volume() for Z in RS])/300
RS[300].volume()

sum([Z.volume() for Z in RS_lu])/300+sum([Z.volume() for Z in RS_ld])/300
RS_lu[300].volume()+RS_ld[300].volume()
'''