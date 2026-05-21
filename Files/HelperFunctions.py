import numpy as np

# Funktioner skal nok ændres ud fra, hvordan EV udregnes

def utility(c, d, par):
    return (c**par.alpha*(d+par.d_floor)**(1-par.alpha))**(1-par.rho) / (1 - par.rho)

def marginalUtility(c, d, par):
    return par.alpha * c ** (par.alpha * (1 - par.rho) - 1) *(d + par.d_floor) ** ((1 - par.alpha) * (1 - par.rho))

def w(expected_value, par):
    return par.beta * expected_value

def q(expected_uc, par):
    return par.beta*par.R*expected_uc

def z(d, q, par):

    numerator = q
    denominator = (par.alpha * (d + par.d_floor)) ** ((1-par.alpha)*(1-par.rho))

    return (numerator / denominator ) ** (1/(par.alpha*(1-par.rho)-1))
