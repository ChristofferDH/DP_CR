import numpy as np
import HelperFunctions as Func


def terminalPeriod(par, sol):

    alpha = par.alpha
    d_floor = par.d_floor
    tau = par.tau

    for i_n, n in enumerate(par.grid_n[par.T-1, :]):
        for i_m, m in enumerate(par.grid_m[par.T-1,:]):
            u_keep = Func.utility(m,n,par) 
            v_keep = u_keep
            uc_keep = Func.marginalUtility(m, n, par)

            x = m + (1 - tau) * n
            c_adjust = alpha * (x + d_floor)
            d_adjust = c_adjust  + (1- alpha)/alpha * c_adjust - d_floor
            u_adjust = Func.utility(c_adjust, d_adjust, par)
            v_adjust = u_adjust
            uc_adjust = Func.marginalUtility(c_adjust, d_adjust, par)
            

            if v_keep >= v_adjust:
                sol.v[par.T-1,:, i_n, i_m] = v_keep
                sol.c[par.T-1,:, i_n, i_m] = m
                sol.d[par.T-1,:, i_n, i_m] = n
                sol.uc[par.T-1,:, i_n, i_m] = uc_keep
            else:
                sol.v[par.T-1,:, i_n, i_m] = v_adjust
                sol.c[par.T-1,:, i_n, i_m] = c_adjust
                sol.d[par.T-1,:, i_n, i_m] = d_adjust
                sol.uc[par.T-1,:, i_n, i_m] = uc_adjust
    sol.m[par.T-1,:, :, :] = 0    

    return sol

def NEGM(t, par, sol):

    sol = EGMUpperEnvelope(t, par, sol, p, d)
    return sol


def EGMUpperEnvelope(t, par, sol, p, d):
    # inputs

    grid_m = par.grid_m
    grid_a = par.grid_a

    # initialize v
    
    v = np.full(len(grid_m), -np.inf)

    # optimal consumption 

    for i_a, a in enumerate(grid_a):
        w[i_a] = w[p,d,i_a]

        q[i_a] = q[p,d,i_a]
        c[i_a] = func.z(d, q[i_a], par)

        m[i_a] = a + c[i_a] 
    
    # borrowing constraint

    for j in range(grid_m):
        if grid_m[j] <= grid_m[1]:
            c[j] = grid_m[j]
            v[j] = func.utility(c[j],d,par) + w[1]

    # Interpolate the optimal consumption on to the exogenous grid

    for i in range(grid_a - 1):
        for j in range(grid_m):
            if m[i] <= grid_m[j] <= m[i+1]:
                c[i,j] = c[i] + (c[i+1]-c[i])/(m[i+1]-m[i])*(m[j]-m[i])
                v[i,j] = func.utility(c[i,j],d) + w[i] + (w[i+1]-w[i])/(a[i+1]-a[i])*((m[j]-c[i,j])-a[i])
                if v[i,j] > v[j]:
                    v[j] = v[i,j]
                    c[j] = c[i,j]
                    
    return c, v

def PostDecisionsFunctions(sol, t, par):
    # inputs
    grid_p = par.grid_p
    grid_n = par.grid_n
    grid_m = par.grid_m
    grid_a = par.grid_a

    grid_psi = par.grid_psi
    grid_zeta = par.grid_zeta
    weight_psi = par.psi_weight
    weight_zeta = par.zeta_weight

    v_next = sol.v[t+1,:,:,:]
    uc_next = sol.uc[t+1,:,:,:]
    m_next = np.zeros(len(grid_a))
    # initialize post decision functions

    shape = (par.T, len(grid_p), len(grid_n), len(grid_a))
    
    w = np.zeros(shape)
    q = np.zeros(shape)

    # Crazy MF loop
    
    for jp in range(grid_p):
        for jn in range (grid_n):
            for jpsi in range(grid_psi):
                for jzeta in range(grid_zeta):
                    p_next = grid_psi[jpsi+1] * grid_p[jp] ** par.Lambda
                    n_next = (1 - par.delta) * grid_n[jn]
                    y_next = p_next * grid_zeta[jzeta]
                    for ja in range(grid_a):
                        m_next[ja] = par.R * grid_a[ja] + y_next
                    
                    v_next = vectorInterpolationKeep(par, p_next, n_next, m_next, v_next)
                    uc_next= vectorInterpolationKeep(par, p_next, n_next, m_next, uc_next)

                    for ja in range(grid_a):
                        w[t, jp, jn, ja] = par.beta * weight_psi[jpsi] * weight_zeta[jzeta] * v_next 
                        q[t, jp, jn, ja] = par.beta * par.R * weight_psi[jpsi] * weight_zeta[jzeta] * uc_next
    return w, q

def vectorInterpolationKeep(par, p, n, m, v):
    grid_p = par.grid_p
    grid_n = par.grid_n
    grid_m = par.grid_m

    jp = np.searchsorted(grid_p, p, side='left') - 1
    jn = np.searchsorted(grid_n, n, side='left') - 1
    jm_vector = np.zeros(len(m))
    for i in range(m):
        if i == 0:
            jm_vector[i] = np.searchsorted(grid_m, m[i], side='left') - 1
        else:
            jm_vector[i] = jm_vector[i-1]
            while m[i] >= grid_m[jm_vector[i] + 1]:
                jm_vector[i] += 1
        
    value_function = np.zeros(len(m))
    binary_array = [0, 1]

    for kp in binary_array:
        if kp == 0:
            omega_p = grid_p[jp + 1] - p
        else:
            p - grid_p[jp]
        for kn in binary_array:
            if kn == 0:
                omega_n = grid_n[jn + 1] - n
            else:
                n - grid_p[jn]

            for i in range(value_function):
                Omega = (grid_p[jp + 1] - grid_p[jp]) * (grid_n[jn + 1] - grid_n[jn]) * (grid_m[jm_vector[i] + 1] - grid_m[jm_vector[i]])
                for km in binary_array:
                    if km == 0:
                        omega_m = grid_m[jm_vector[i] + 1] - m[i]
                    else:
                        omega_m = m[i] - grid_m[jm_vector[i]]
                    value_function[i] += (omega_p * omega_n * omega_m)/Omega * v[:, jp + kp, jn +kn, jm_vector[i] + km]
    return value_function  