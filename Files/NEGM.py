import numpy as np
from Files import HelperFunctions as Function

def NEGMalg(par, sol):

    #Step 0: Solving the terminal period

    sol = terminalPeriod(par, sol)

    grid_n = par.grid_n
    grid_m = par.grid_m

    v_temp = np.full((par.T - 1, par.p_N, par.n_N, par.n_N, par.m_N), np.nan) #Temporary array for the adjustment problem
    c_temp = np.full((par.T - 1, par.p_N, par.n_N, par.n_N, par.m_N), np.nan)

    c_keep = np.full((par.T - 1, par.p_N, par.n_N, par.m_N), np.nan)
    v_keep = np.full((par.T - 1, par.p_N, par.n_N, par.m_N), np.nan)

    c_adjust = np.full((par.T - 1, par.p_N, par.n_N, par.m_N), np.nan)
    v_adjust = np.full((par.T - 1, par.p_N, par.n_N, par.m_N), np.nan)
    d_adjust = np.full((par.T - 1, par.p_N, par.n_N, par.m_N), np.nan)

    
    for t in range(par.T - 2, -1, -1):
        print(t)
        # Step 1: Finding w and q
        w, q = postDecisionFunctions(sol, t, par)

        # Step 2: Solving the keeper problem for c and v
        for jp in range(par.p_N):
            for jd, d in enumerate(grid_n[t,:]):
                c_keep[t, jp, jd, :], v_keep[t, jp, jd, :] = EGMUpperEnvelope(t, par, w, q, jp, jd)
        
        # Step 3: solving the adjustment problem
        for jm, m in enumerate(grid_m[t, :]):        
            for jp in range(par.p_N):
                for jn, n in enumerate(grid_n[t, :]):
                    x = m + (1-par.tau)*n
                    for jd, d in enumerate(grid_n[t, :]): 
                        if x >= d:
                            m_new = x - d
                            v_temp[t, jp, jn, jd, jm] = np.interp(m_new, grid_m[t, :], v_keep[t, jp, jd, :])
                            c_temp[t, jp, jn, jd, jm] = np.interp(m_new, grid_m[t, :], c_keep[t, jp, jd, :])
                        else:
                            v_temp[t, jp, jn, jd, jm] = -1e12
                            c_temp[t, jp, jn, jd, jm] = -1e12
                    d_opt = np.argmax(v_temp[t, jp, jn, :, jm])
                    d_adjust[t, jp, jn, jm] = grid_n[t, d_opt]
                    v_adjust[t, jp, jn, jm] = v_temp[t, jp, jn, d_opt, jm]
                    c_adjust[t, jp, jn, jm] = c_temp[t, jp, jn, d_opt, jm]
                  
        #Step 4: comparing the two solutions and choosing the better one             
        for jp in range(par.p_N):
            for jn in range(par.n_N):
                sol.m[t, jp, jn, :] = grid_m[t, :]
                for jm in range(par.m_N):
                    if v_keep[t, jp, jn, jm] >= v_adjust[t, jp, jn, jm]:
                        sol.v[t, jp, jn, jm] = v_keep[t, jp, jn, jm]
                        sol.c[t, jp, jn, jm] = c_keep[t, jp, jn, jm]
                        sol.d[t, jp, jn, jm] = grid_n[t, jn]
                            
                    else:
                        sol.v[t, jp, jn, jm] = v_adjust[t, jp, jn, jm]
                        sol.c[t, jp, jn, jm] = c_adjust[t, jp, jn, jm]
                        sol.d[t, jp, jn, jm] = d_adjust[t, jp, jn, jm]

        sol.uc[t,:,:,:] = Function.marginalUtility(sol.c[t,:,:,:], sol.d[t,:,:,:], par)

    return sol   

def terminalPeriod(par, sol):

    alpha = par.alpha
    d_floor = par.d_floor
    tau = par.tau

    for i_n, n in enumerate(par.grid_n[par.T-1, :]):
        for i_m, m in enumerate(par.grid_m[par.T-1,:]):
            u_keep = Function.utility(m,n,par)
            v_keep = u_keep
            uc_keep = Function.marginalUtility(m, n, par)

            x = m + (1 - tau) * n
            c_adjust = alpha * (x + d_floor)
            d_adjust = (1- alpha)/alpha * c_adjust - d_floor
            u_adjust = Function.utility(c_adjust, d_adjust, par)
            if x >= alpha/(1 - alpha) * d_floor:
                v_adjust = u_adjust
                uc_adjust = Function.marginalUtility(c_adjust, d_adjust, par)
            else:
                v_adjust = -np.inf
                uc_adjust = np.nan
            
            sol.m[par.T-1,:, i_n, i_m] = m
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
    sol.a[par.T-1,:, :, :] = 0    

    return sol
  

def EGMUpperEnvelope(t, par, w, q, jp, jd):
    
    # inputs

    grid_m = par.grid_m[t,:]
    grid_a = par.grid_a[t,:]
    d = par.grid_n[t, jd]

    # initialize
    
    v = np.full(len(grid_m), -np.inf)
    c = np.full(len(grid_m), np.nan)
    ci = np.full(len(grid_a), np.nan)
    mi = np.full(len(grid_a), np.nan)

    # optimal consumption 

    wi = w[jp, jd, :]
    qi = q[jp, jd, :]

    for i_a, a in enumerate(grid_a):

        ci[i_a] = Function.z(d, qi[i_a], par)

        mi[i_a] = a + ci[i_a]
    # borrowing constraint

    for j, m in enumerate(grid_m):
        if m <= mi[0]:
            c[j] = m
            v[j] = Function.utility(c[j],d,par) + wi[0]

    # Interpolate the optimal consumption on to the exogenous grid

    for i in range(len(grid_a) - 1):
            for j, m in enumerate(grid_m):
                if mi[i] <= m <= mi[i+1]:
                    c_ij = ci[i] + (ci[i+1]-ci[i])/(mi[i+1]-mi[i])*(m-mi[i])
                    v_ij = Function.utility(c_ij, d, par) + wi[i] + (wi[i+1]-wi[i])/(grid_a[i+1]-grid_a[i])*((m-c_ij)-grid_a[i])
                    if v_ij > v[j]:
                        v[j] = v_ij
                        c[j] = c_ij
                    
    return c, v

def postDecisionFunctions(sol, t, par):
    # inputs
    grid_p = par.grid_p[t,:]
    grid_n = par.grid_n[t,:]
    grid_a = par.grid_a[t,:]

    v_next = sol.v[t+1,:,:,:] 
    uc_next = sol.uc[t+1,:,:,:]
    
    # initialize post decision functions

    shape = (len(grid_p), len(grid_n), len(grid_a))
    
    w = np.zeros(shape)
    q = np.zeros(shape)

    # Crazy MF loop
    for jp in range(len(grid_p)):
        for jn in range (len(grid_n)):
            for jshock in range(par.number_of_shocks):
                psi  = par.psi_vec[jshock]
                zeta = par.zeta_vec[jshock]
                weight = par.shock_weight[jshock]

                p_next = psi * grid_p[jp] ** par.Lambda
                n_next = (1 - par.delta) * grid_n[jn]
                y_next = p_next * zeta
                m_next = par.R * grid_a + y_next
                    
                v_next_interp = vectorInterpolationKeep(par, p_next, n_next, m_next, v_next, t)
                uc_next_interp = vectorInterpolationKeep(par, p_next, n_next, m_next, uc_next, t)

                w[jp, jn, :] += par.beta * weight * v_next_interp
                q[jp, jn, :] += par.beta * par.R * weight * uc_next_interp
                
    return w, q

def vectorInterpolationKeep(par, p, n, m, v, t):
    grid_p = par.grid_p[t,:]
    grid_n = par.grid_n[t,:]
    grid_m = par.grid_m[t,:]

    jp = np.searchsorted(grid_p, p, side='left') - 1
    jn = np.searchsorted(grid_n, n, side='left') - 1
    jm_vector = np.zeros(len(m), dtype = int)

    jp = np.clip(jp, 0, len(grid_p) - 2)
    jn = np.clip(jn, 0, len(grid_n) - 2)
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
            omega_p = p - grid_p[jp]
        for kn in binary_array:
            if kn == 0:
                omega_n = grid_n[jn + 1] - n
            else:
                omega_n = n - grid_n[jn]

            for i in range(len(interp_function)):
                jm_vector[i] = np.clip(jm_vector[i], 0, len(grid_m)-2)
                Omega = (grid_p[jp + 1] - grid_p[jp]) * (grid_n[jn + 1] - grid_n[jn]) * (grid_m[jm_vector[i] + 1] - grid_m[jm_vector[i]])
                for km in binary_array:
                    if km == 0:
                        omega_m = grid_m[jm_vector[i] + 1] - m[i]
                    else:
                        omega_m = m[i] - grid_m[jm_vector[i]]
                    interp_function[i] += (omega_p * omega_n * omega_m)/Omega * v[jp + kp, jn + kn, jm_vector[i] + km]
                    
    return interp_function  