import numpy as np
from Files import HelperFunctions as Func


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

def NEGMalg(t, par, sol):

    sol = EGMUpperEnvelope(t, par, sol, p, d)
    return sol


def EGMUpperEnvelope(t, par, w, q, jp, jd):
    # inputs

    grid_m = par.grid_m[t,:]
    grid_a = par.grid_a[t,:]
    d = par.grid_n[jd]
    # initialize
    
    v = np.full(len(grid_m), -np.inf)

    c = np.full(len(grid_m), np.nan)

    wi = np.full(len(grid_a), np.nan)
    qi = np.full(len(grid_a), np.nan)
    ci = np.full(len(grid_a), np.nan)
    mi = np.full(len(grid_a), np.nan)

    # optimal consumption 

    w_fixpd = w[jp, jd, :]
    q_fixpd = q[jp, jd, :]

    for i_a, a in enumerate(grid_a):
        wi[i_a] = w_fixpd[i_a]
        qi[i_a] = q_fixpd[i_a]

        ci[i_a] = Func.z(d, qi[i_a], par)

        mi[i_a] = a + ci[i_a] 
    
    # borrowing constraint

    for j, m in enumerate(grid_m):
        if m <= mi[0]:
            c[j] = m
            v[j] = Func.utility(c[j],d,par) + wi[0]

    # Interpolate the optimal consumption on to the exogenous grid

    for i in range(len(grid_a) - 1):
        for j, m in enumerate(grid_m):
            if mi[i] <= m <= mi[i+1]:
                c_ij = ci[i] + (ci[i+1]-ci[i])/(mi[i+1]-mi[i])*(m-mi[i])
                v_ij = Func.utility(c_ij, d, par) + wi[i] + (wi[i+1]-wi[i])/(grid_a[i+1]-grid_a[i])*((m-c_ij)-grid_a[i])
                if v_ij > v[j]:
                    v[j] = v_ij
                    c[j] = c_ij
                    
    return c, v

def PostDecisionsFunctions(sol, t, par):
    # inputs
    grid_p = par.grid_p[t,:]
    grid_n = par.grid_n[t,:]
    grid_m = par.grid_m[t,:]
    grid_a = par.grid_a[t,:]

    grid_psi = par.psi_vec
    grid_zeta = par.zeta_vec
    weight_psi = par.psi_weight_vec
    weight_zeta = par.zeta_weight_vec

    v_next = sol.v[t+1,:,:,:] 
    uc_next = sol.uc[t+1,:,:,:]
    m_next = np.zeros(len(grid_a))
    # initialize post decision functions

    shape = (len(grid_p), len(grid_n), len(grid_a))
    
    w = np.zeros(shape)
    q = np.zeros(shape)

    # Crazy MF loop
    
    for jp in range(len(grid_p)):
        for jn in range (len(grid_n)):
            for jpsi in range(len(grid_psi)):
                for jzeta in range(len(grid_zeta)):
                    p_next = grid_psi[jpsi] * grid_p[jp] ** par.Lambda
                    n_next = (1 - par.delta) * grid_n[jn]
                    y_next = p_next * grid_zeta[jzeta]
                    for ja in range(len(grid_a)):
                        m_next[ja] = par.R * grid_a[ja] + y_next
                    
                    v_next_interp = vectorInterpolationKeep(par, p_next, n_next, m_next, v_next, t)
                    uc_next_interp = vectorInterpolationKeep(par, p_next, n_next, m_next, uc_next, t)

                    for ja in range(len(grid_a)):
                        w[jp, jn, ja] = par.beta * weight_psi[jpsi] * weight_zeta[jzeta] * v_next_interp[ja]
                        q[jp, jn, ja] = par.beta * par.R * weight_psi[jpsi] * weight_zeta[jzeta] * uc_next_interp[ja]
    return w, q

def vectorInterpolationKeep(par, p, n, m, v, t):
    grid_p = par.grid_p[t,:]
    grid_n = par.grid_n[t,:]
    grid_m = par.grid_m[t,:]

    jp = np.searchsorted(grid_p, p, side='left') - 1
    jn = np.searchsorted(grid_n, n, side='left') - 1
    jm_vector = np.zeros(len(m), dtype = int)

    jp = min(jp, len(grid_p) - 2)
    jn = min(jn, len(grid_n) - 2)
    for i in range(len(m)):
        if i == 0:
            jm_vector[i] = np.searchsorted(grid_m, m[i], side='left') - 1
        else:
            jm_vector[i] = jm_vector[i-1]
            while jm_vector[i] + 1 < len(grid_m) and m[i] >= grid_m[jm_vector[i] + 1]:
                jm_vector[i] += 1
        
    interp_function = np.zeros(len(m))
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

            for i in range(len(interp_function)):
                jp = min(jp, len(grid_p) - 2)
                jn = min(jn, len(grid_n) - 2)
                jm_vector[i] = min(jm_vector[i], len(grid_m) - 2)
                Omega = (grid_p[jp + 1] - grid_p[jp]) * (grid_n[jn + 1] - grid_n[jn]) * (grid_m[jm_vector[i] + 1] - grid_m[jm_vector[i]])
                for km in binary_array:
                    if km == 0:
                        omega_m = grid_m[jm_vector[i] + 1] - m[i]
                    else:
                        omega_m = m[i] - grid_m[jm_vector[i]]
                    interp_function[i] += (omega_p * omega_n * omega_m)/Omega * v[jp + kp, jn + kn, jm_vector[i] + km]
    return interp_function  