import numpy as np
import HelperFunctions as func


def EGMUpperEnvelope(sol, t, par, w, q, i_p, i_d):
    # EGM and upper envelope

    grid_m = par.grid_m
    grid_a = par.grid_a
    
    value_function = np.full(len(grid_m), -np.inf)

    for i_a,a in enumerate(grid_a):
        w[i_a] = w[i_p,i_d,i_a]

        q[i_a] = q[i_p,i_d,i_a]
        c[i_a] = func.z(i_d, q[i_a], par)

        m[i_a] = a + c[i_a] 
    
    for j in range(grid_m):
        if grid_m[j] <= m[0]
            c[j] = grid_m[j]
            value_function[j] = func.utility(c[j],i_d,par) + w[0]

    for i in range(grid_a - 1):
        for j in range(grid_m):
            if m[j] 


    return c, value_function

def PostDecisionsFunctions(t, par):
    grid_p = par.grid_p
    grid_n = par.grid_n
    grid_m = par.grid_m
    grid_a = par.grid_a

    grid_psi = par.grid_psi
    grid_zeta = par.grid_zeta
    weight_psi = par.psi_weight
    weight_zeta = par.zeta_weight

    v_next
    uc_next
    shape = (len(grid_p), len(grid_n), len(grid_a))
    
    w = np.zeros(shape)
    q = np.zeros(shape)
    for jp in range(grid_p):
        for jn in range (grid_n):
            for jpsi in range(grid_psi):
                for jzeta in range(grid_zeta):
                    p_next = grid_psi[jpsi+1] * grid_p[jp] ** par.Lambda
                    n_next = (1 - par.delta) * grid_n[jn]
                    y_next = p_next * grid_zeta[jzeta]
                    for ja in range(grid_a):
                        m_next = par.R * grid_a[ja] + y_next
                    
                    v_next = magic.function(nemt)
                    uc_next = magic.function(nemt)

                    for ja in range(grid_a):
                        w[noget +1] = weight_psi[jpsi] * weight_zeta[jzeta] * v_next 
                        q[noget +1] = weight_psi[jpsi] * weight_zeta[jzeta] * uc_next
    return w, q

def vectorInterpolation(par, p, n, m):
    grid_p = par.grid_p
    grid_n = par.grid_n
    grid_m = par.grid_m
    grid_a = par.grid_a   

    jp = grid_p.index(p)
    jn = grid_p.index(n)