"""Evaluate the displayed comparison constants; no dynamical sampling."""
from pathlib import Path
import json
import sympy as sp

omega = sp.Integer(1)
sigma = sp.Integer(2)
gamma = sigma**2 / 4
a_star = 2 * sp.sqrt(omega**2 + 4)
r_star = 2 * abs(sigma)
f_star = sigma**2
g_star = abs(sigma)**3
k_hat = a_star + f_star + g_star
C_T4 = (a_star**2 / 2
        + (3*a_star*r_star**2 + 3*a_star**2*r_star + a_star**3)/6
        + (a_star+r_star)**4/24)
C_4 = (C_T4 + f_star*(a_star+(a_star+r_star)**2/2)
       + g_star*(a_star+r_star) + k_hat**2/2 + (f_star+g_star)*k_hat)
M_gamma = 1/gamma + gamma/(4*omega**2) + sp.sqrt(gamma**2+4*omega**2)/(4*omega**2)
C_A = sp.sqrt((1+(1+4/omega**2)**2)/2)
c_end = sp.sqrt(2)*C_A*(f_star+g_star) + 4/omega
c_time = C_A/sp.sqrt(2)*(C_4+2*gamma*M_gamma**2*g_star**2)

# The endpoint estimate is for ||X||_F=|a|/sqrt(2).
# Each of two final partial cells costs at most
# (sqrt(2)/omega)*epsilon*|a_k|*||X||_F, with |a_k|<=2.
assert sp.simplify(c_end - (2*C_A*(f_star+g_star)/sp.sqrt(2) + 4/omega)) == 0
assert all(x.is_positive for x in [C_T4, C_4, M_gamma, C_A, c_end, c_time])

N, amplitude, horizon = sp.Integer(32), sp.Rational(1,4), sp.Integer(1)
epsilon = N**-3
mode_error_bound = amplitude*epsilon*(c_end+c_time*N**2*horizon)
D = (3-sp.sqrt(5))/2
kappa = D/(2*gamma)
quartic = kappa/12+kappa**2*(gamma/(4*sp.sqrt(5))-1/(2*gamma))
quartic_rate_signal = abs(quartic)*(2*sp.pi/N)**4

quantities = dict(C_T4=C_T4,C_4=C_4,M_gamma=M_gamma,C_A=C_A,
                  c_end=c_end,c_time=c_time,
                  momentum_law_mode_bound=mode_error_bound,
                  initial_mode_amplitude=amplitude/2,
                  quartic_stochastic_rate_signal=quartic_rate_signal,
                  epsilon=epsilon)
out = {
    'scope': 'Evaluation of the analytic upper bound and stochastic Taylor signal; no trajectory data and no numerical test of either deterministic theorem.',
    'parameters': {'omega0':1,'sigma':2,'gamma':1,'N':32,'k':1,'a':'1/4','T':1,'epsilon':'N^-3'},
    'values': {key:{'exact':str(sp.simplify(value)),
                    'decimal_50':str(sp.N(value,50))} for key,value in quantities.items()},
    'interpretation': 'The momentum-law comparison bound exceeds the initial mode amplitude at these parameters. It gives no useful finite-size resolution guarantee. This is not evidence that the actual dynamical discrepancy is large.'
}
dest = Path(__file__).with_suffix('.json')
dest.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
