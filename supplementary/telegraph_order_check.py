"""Exact secular recurrence through eighth order and independent matrix check.

No trajectories or Monte Carlo. The exact example has r=1/2,
omega0^2=1/2, gamma=1. Mathematical interpretation is in Appendix C.
"""
from pathlib import Path
import json
import math
import sympy as s
import numpy as np
import scipy.linalg as la
from independent_scalar_matrix_audit import relative_coordinate_matrix

a=s.Symbol('a',positive=True)
c=s.Rational(5,2); g=s.Integer(1); h=s.Rational(3,2)
x=(c-a)/2; u=1-x*x
size=5
def add(v,w):return [s.expand(v[i]+w[i]) for i in range(size)]
def scale(v,t):return [s.expand(t*v[i]) for i in range(size)]
def mul(v,w):return [s.expand(sum(v[j]*w[i-j] for j in range(i+1))) for i in range(size)]
def inv(v):
    out=[1/v[0]]+[s.Integer(0)]*4
    for i in range(1,size):out[i]=s.expand(-sum(v[j]*out[i-j] for j in range(1,i+1))/v[0])
    return out
cc=s.Symbol('c',positive=True)
moments={-1:1/s.sqrt(cc*cc-4)}
for j in range(2,6):moments[-j]=s.diff(moments[-j+1],cc)/(-j+1)
def average(expr):
    value=0
    for term in s.Add.make_args(s.expand(expr)):
        power=int(term.as_powers_dict().get(a,0))
        coefficient=s.cancel(term/a**power)
        assert a not in coefficient.free_symbols
        if power<0:moment=moments[power].subs(cc,c)
        else:moment=sum(s.binomial(power,2*j)*c**(power-2*j)*s.binomial(2*j,j) for j in range(power//2+1))
        value+=coefficient*moment
    return s.simplify(value)
z=[s.Integer(0)]*size
for n in range(1,size):
    rz=add(scale(z,g),mul(z,z))
    S=[2*a,x/2,-x/96,x/11520,0]
    d2=[0,4*u,-u/3,u/90,-u/5040]
    denom=scale(mul(add([g,0,0,0,0],z),add(S,rz)),2*g)
    E=add(scale(z,1/g),mul(add(d2,scale(mul(rz,rz),-1)),inv(denom)))
    E2=mul(E,E); E3=mul(E2,E); E4=mul(E3,E)
    residual=add(add(E,scale(E2,-1)),add(E3,scale(E4,-1)))
    z[n]=-g*average(residual[n])
kappa=-z[1]
M=s.Rational(1,12)-z[2]/kappa
eta=s.Symbol('eta')
q=eta+eta**2/12+eta**3/90+eta**4/560
mu=s.series(sum(z[j]*q**j for j in range(1,5)),eta,0,5).removeO()
telegraph=s.series(mu+(M/kappa)*mu**2+kappa*eta,eta,0,5).removeO().expand()
assert all(s.simplify(telegraph.coeff(eta,j))==0 for j in range(1,4))
defect=s.factor(telegraph.coeff(eta,4))
assert z[1:]==[-s.Rational(1,4),s.Integer(0),-s.Rational(1,1440),-s.Rational(127,1161216)]
assert defect==-s.Rational(1,165888)

def matrix_check(n):
    L,v,left,minus,plus=relative_coordinate_matrix(n,math.sqrt(.5),1.)
    d=4*n; border=np.zeros((d+1,d+1))
    border[:d,:d]=L; border[:d,d]=-v; border[d,:d]=left
    factors=la.lu_factor(border); vectors=[v]; lambdas=[0.]
    for degree in range(1,9):
        rhs=np.zeros(d)
        for j in range(1,degree+1):
            shift=(minus+(-1)**j*plus)/math.factorial(j)
            rhs[2*n:3*n]-=shift@vectors[degree-j][:n]
            rhs[3*n:]-=shift@vectors[degree-j][n:2*n]
        for j in range(1,degree):rhs+=lambdas[j]*vectors[degree-j]
        result=la.lu_solve(factors,np.r_[rhs,0.])
        vectors.append(result[:d]); lambdas.append(result[d])
    coeff=[((-1)**j)*lambdas[2*j] for j in range(1,5)]
    errors=[abs(coeff[j-1]-float(z[j])) for j in range(1,5)]
    if n==64:assert max(errors)<1e-10
    return {'N':n,'xi_coefficients':coeff,'absolute_differences':errors}

w=s.Symbol('w',positive=True)
hh=w*s.sqrt(w*w+4); rr=(w*w+2-hh)/2
weak={
 'gamma_B_over_2sqrtomega':str(s.series(s.sqrt(6*rr*hh/(2*hh+3*rr))/(2*s.sqrt(w)),w,0,3)),
 'gamma_M_over_2sqrtomega':str(s.series(s.sqrt(2*hh)/(2*s.sqrt(w)),w,0,3))}
report={'model':{'omega0_squared':'1/2','gamma':'1','r':'1/2'},
 'exact_xi_coefficients':{str(2*j):str(z[j]) for j in range(1,5)},
 'M':str(M),'eta4_coefficient':str(mu.expand().coeff(eta,4)),
 'telegraph_eta4_defect':str(defect),'matrix_checks':[matrix_check(n) for n in [32,64]],'weak_pinning':weak,
 'scope':'Exact algebraic integral recurrence; floating-point matrix comparisons are arithmetic checks, not rigorous numerical enclosures.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
