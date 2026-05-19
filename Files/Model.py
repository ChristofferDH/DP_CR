import numpy as np
from types import SimpleNamespace
import Tools as Tools
import NEGM

class Durable_BufferStock():
    
    def __init__(self, name = None):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()

        
    def modelSetup(self):
        par = self.par

        # All of these parameters are taken from Druedahl (2020)
        par.T = 10
        par.beta = 0.965
        par.rho = 2
        par.alpha = 0.9
        par.d_floor = 0.01
        par.R = 1.03
        par.tau = 0.1
        par.delta = 0.15
        par.sigma_psi = 0.1 # Permanent
        par.sigma_zeta = 0.1 # Transitory
        par.Lambda = 1 #Persistency parameter

        # Grids!!!
        # pre and post-decisision state grid
        ### p: grid settings
        if 0:
            par.p_min = 0.0001
            par.p_max = 3
            par.p_N = 150 
        
        ## Post-decision state grids

        ### d: grid settings
        
            ### a: grid settings
            par.a_min = 0.0001
            par.a_max = 11
            par.a_N = 300 

            ## Pre-decision state grids
            ### x: grid settings
            par.x_min = 0.0001
            par.x_max = 13
            par.x_N = 150 

            ### n: grid settings
            par.n_min = 0.0001
            par.n_max = 3
            par.n_N = 150 

            ### m: grid settings
            par.m_min = 0.0001
            par.m_max = 10
            par.m_N = 300 

        par.p_min = 0.0001
        par.p_max = 3
        par.p_N = 20
        
        ### a: grid settings
        par.a_min = 0.0001
        par.a_max = 11
        par.a_N = 30 

            ## Pre-decision state grids
            ### x: grid settings
        par.x_min = 0.0001
        par.x_max = 13
        par.x_N = 30 

            ### n: grid settings
        par.n_min = 0.0001
        par.n_max = 3
        par.n_N = 10 

            ### m: grid settings
        par.m_min = 0.0001
        par.m_max = 10
        par.m_N = 30
        
        # Numerical integration
        ## Shock grid settings
        par.N_psi = 5
        par.N_zeta = 5

    def create_grids(self):
        par = self.par

        # State grids

        par.grid_a = np.nan + np.zeros((par.T, par.a_N))
        par.grid_x = np.nan + np.zeros((par.T, par.x_N))
        par.grid_n = np.nan + np.zeros((par.T, par.n_N))
        par.grid_m = np.nan + np.zeros((par.T, par.m_N))
        par.grid_p = np.nan + np.zeros((par.T, par.p_N))
        
        for t in range(par.T):
            par.grid_a[t,:] = Tools.gridFunction(par.a_min,par.a_max, par.a_N)
            par.grid_x[t,:] = Tools.gridFunction(par.x_min,par.x_max, par.x_N)
            par.grid_n[t,:] = Tools.gridFunction(par.n_min,par.n_max, par.n_N)
            par.grid_m[t,:] = Tools.gridFunction(par.m_min,par.m_max, par.m_N)
            par.grid_p[t,:] = Tools.gridFunction(par.p_min,par.p_max, par.p_N)

        # Quadrature: nodes and weights

        par.psi, par.psi_weight = Tools.GaussHermite_lognorm(par.sigma_psi,par.N_psi)
        par.zeta,par.zeta_weight = Tools.GaussHermite_lognorm(par.sigma_zeta,par.N_zeta)

        par.psi_vec = np.tile(par.psi,par.zeta.size) 
        par.zeta_vec = np.repeat(par.zeta,par.psi.size)  
        par.psi_weight_vec = np.tile(par.psi_weight,par.zeta.size)
        par.zeta_weight_vec = np.repeat(par.zeta_weight,par.psi.size)

        par.shock_weight = par.psi_weight_vec * par.zeta_weight_vec
        assert (1-sum(par.shock_weight) < 1e-8), 'The weights do not sum to 1'
        par.number_of_shocks = par.shock_weight.size    # count number of shock nodes

    def solve(self):

        # initialize
        sol = self.sol
        par = self.par

        shape = (par.T,par.p_N,par.n_N,par.m_N)
        sol.v = np.nan + np.zeros(shape)
        sol.uc = np.nan + np.zeros(shape)
        sol.c = np.nan + np.zeros(shape)
        sol.d = np.nan + np.zeros(shape)
        sol.m = np.nan + np.zeros(shape)

        # Terminal period
        sol.v[par.T-1,:,:,:]
        sol.uc[par.T-1,:,:,:]
        sol.c[par.T-1,:,:,:]
        sol.d[par.T-1,:,:,:]
        sol.m[par.T-1,:,:,:] 

        #for t in range(par.T - 2, -1, -1):
            #NEGM.NEGM(t, par, sol)

        
        # step 1: compute post-decision functions (algorithm 5)

       # w, q = NEGM.PostDecisionsFunctions(sol, t, par)

        # step 2: solve the keeper problem (algorithm 1)



        # step 3: solve the adjuster problem. Interpolation of v_keep

