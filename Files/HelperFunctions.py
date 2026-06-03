def utility(c, d, par):
    if par.CobbDouglas:
        return (c**par.alpha*(d+par.d_floor)**(1-par.alpha))**(1-par.rho) / (1 - par.rho)
    else:
        sigma = par.sigma
        first_term = par.alpha ** (1/sigma) * c ** ((sigma - 1) / sigma)
        second_term = (1 - par.alpha) ** (1/sigma) * (d + par.d_floor) ** ((sigma - 1) / sigma)
        return (first_term + second_term) ** (sigma/(sigma-1))

def marginalUtility(c, d, par): 
    if par.CobbDouglas:
        return par.alpha * c ** (par.alpha * (1 - par.rho) - 1) *(d + par.d_floor) ** ((1 - par.alpha) * (1 - par.rho))
    else:
        sigma = par.sigma
        first_term = par.alpha ** (1/sigma) * c ** ((sigma - 1) / sigma)
        second_term = (1 - par.alpha) ** (1/sigma) * (d + par.d_floor) ** ((sigma - 1) / sigma)
        return par.alpha ** (1/sigma) * c ** (-1/sigma) * (first_term + second_term) ** (1/(sigma - 1))

def z(d, q, par):
    
    if par.CobbDouglas:
        numerator = q
        denominator = par.alpha * (d + par.d_floor) ** ((1-par.alpha)*(1-par.rho))
        return (numerator / denominator ) ** (1/(par.alpha*(1-par.rho)-1))
    else:
        numerator = (1 - par.alpha) ** (1-par.sigma) * (par.d_floor + d) ** ((par.sigma-1)/par.sigma)
        denominator = (q/(par.alpha ** (1/par.sigma))) ** (par.sigma - 1) - par.alpha ** (1 / par.sigma)
        return (numerator / denominator) ** (par.sigma / (par.sigma - 1))
