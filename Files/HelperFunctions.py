def utility(c, d, par, CobbDouglas = True):
    if CobbDouglas:
        return (c**par.alpha*(d+par.d_floor)**(1-par.alpha))**(1-par.rho) / (1 - par.rho)
    else:
        sigma = par.sigma
        first_term = par.alpha ** (1/sigma) * c ** ((sigma - 1) / sigma)
        second_term = (1 - par.alpha) ** (1/sigma) * (d + par.d_floor) ** ((sigma - 1) / sigma)
        return (first_term + second_term) ** (sigma/(sigma-1))

def marginalUtility(c, d, par, CobbDouglas = True):
    if CobbDouglas:
        return par.alpha * c ** (par.alpha * (1 - par.rho) - 1) *(d + par.d_floor) ** ((1 - par.alpha) * (1 - par.rho))
    else:
        pass

def z(d, q, par, CobbDouglas = True):
    numerator = q
    if CobbDouglas:
        denominator = (par.alpha * (d + par.d_floor)) ** ((1-par.alpha)*(1-par.rho))
        return (numerator / denominator ) ** (1/(par.alpha*(1-par.rho)-1))
    else:
        pass
