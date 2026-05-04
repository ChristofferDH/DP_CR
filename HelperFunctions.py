import numpy as np

# Funktioner skal nok ændres ud fra, hvordan EV udregnes

def w(EV, par):
    return par.beta * EV

def q(EV, par):
    return par.beta*par.R*EV

def z(d, q, par):

    numerator = q
    denominator = (par.alpha * (d + par.d_floor)) ** ((1-par.alpha)*(1-par.rho))

    return (numerator / denominator ) ** (1/(par.alpha*(1-par.rho)-1))
