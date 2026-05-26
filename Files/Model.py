import numpy as np
from types import SimpleNamespace
from Files import Tools as Tools
from Files import NEGM
from time import process_time

class DurableBufferStock():
    
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
        par.p_min = 0.0001
        par.p_max = 3
        par.p_N = 40
        
        ### a: grid settings
        par.a_min = 0.0001
        par.a_max = 11
        par.a_N = 80 

        ## Pre-decision state grids
        
        ### n: grid settings
        par.n_min = 0.0001
        par.n_max = 3
        par.n_N = 100 

        ### m: grid settings
        par.m_min = 0.0001
        par.m_max = 10
        par.m_N = 80

        ### x: grid settings
        par.x_min = 0.0001
        par.x_max = par.m_max + par.n_max
        par.x_N = 80 
        
        # Numerical integration
        ## Shock grid settings
        par.N_psi = 5
        par.N_zeta = 5

        # Simulation
        par.simN = 1000 # number of persons in simulation
        par.sim_m_ini = 2.5 # initial m in simulation
        par.sim_p_ini = 1.0 # initial p in simulation
        par.sim_n_ini = 1.0 # initial n in simulation

    def createGrids(self):
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
        par.psi, par.psi_weight = Tools.GaussHermiteLognorm(par.sigma_psi,par.N_psi)
        par.zeta,par.zeta_weight = Tools.GaussHermiteLognorm(par.sigma_zeta,par.N_zeta)

        par.psi_vec = np.tile(par.psi,par.zeta.size) 
        par.zeta_vec = np.repeat(par.zeta,par.psi.size)  
        par.psi_weight_vec = np.tile(par.psi_weight,par.zeta.size)
        par.zeta_weight_vec = np.repeat(par.zeta_weight,par.psi.size)

        par.shock_weight = par.psi_weight_vec * par.zeta_weight_vec
        assert (1-sum(par.shock_weight) < 1e-8), 'The weights do not sum to 1'
        par.number_of_shocks = par.shock_weight.size    # count number of shock nodes

    def solve(self):
        tic = process_time()
        # initialize

        sol = self.sol
        par = self.par

        shape = (par.T, par.p_N, par.n_N, par.m_N)
        sol.v = np.nan + np.zeros(shape)
        sol.uc = np.nan + np.zeros(shape)
        sol.c = np.nan + np.zeros(shape)
        sol.d = np.nan + np.zeros(shape)
        sol.m = np.nan + np.zeros(shape)
        sol.a = np.nan + np.zeros(shape)
        sol.n = np.nan + np.zeros(shape)
        sol.v_keep = np.nan + np.zeros(shape)
        sol.v_adjust = np.nan + np.zeros(shape)

        # Run the MF algorithm and pray
        print("Number of gridpoints:", "t =", par.T, "p =", par.p_N, "n =", par.n_N, "m =", par.m_N, )
        sol = NEGM.NEGMalg(par, sol)

        toc = process_time()
        print(f'Solver time: {toc-tic:.2f} seconds')



    def simulate(self):
        tic = process_time()

        par = self.par
        sol = self.sol
        sim = self.sim

        shape = (par.T, par.simN)

        sim.v_keep = np.nan + np.zeros(shape)
        sim.v_adjust = np.nan + np.zeros(shape)
        sim.c = np.nan + np.zeros(shape)
        sim.d = np.nan + np.zeros(shape)
        sim.m = np.nan + np.zeros(shape)
        sim.p = np.nan + np.zeros(shape)
        sim.n = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.y = np.nan + np.zeros(shape)

            
        shocki = np.random.choice(par.number_of_shocks,(par.T,par.simN),replace=True,p=par.shock_weight) 
        sim.psi = par.psi_vec[shocki] 
        sim.zeta = par.zeta_vec[shocki]
            
        #check it has a mean of 1
        assert (abs(1-np.mean(sim.psi)) < 1e-2), 'The mean is not 1 in the simulation of xi'
        assert (abs(1-np.mean(sim.zeta)) < 1e-2), 'The mean is not 1 in the simulation of psi'

        # Initial values
        sim.p[0,:] = par.sim_p_ini
        sim.n[0,:] = par.sim_n_ini
        sim.m[0,:] = par.sim_m_ini

        # Simulation
        for t in range(par.T):
            sim.c[t,:] = NEGM.LinearInterp(par, sim.p[t,:], sim.n[t,:], sim.m[t,:], sol.c[t,:,:,:], t)
            sim.d[t,:] = NEGM.LinearInterp(par, sim.p[t,:], sim.n[t,:], sim.m[t,:], sol.d[t,:,:,:], t)
            sim.v_keep[t,:] = NEGM.LinearInterp(par, sim.p[t,:], sim.n[t,:], sim.m[t,:], sol.v_keep[t,:,:,:], t)
            sim.v_adjust[t,:] = NEGM.LinearInterp(par, sim.p[t,:], sim.n[t,:], sim.m[t,:], sol.v_adjust[t,:,:,:], t)
            
            for i in range(par.simN):
                if sim.v_keep[t,i] >= sim.v_adjust[t,i]:
                    sim.a[t,i] = sim.m[t,i] - sim.c[t,i]
                else:
                    sim.a[t,i] = sim.m[t,i] + (1-par.tau) * sim.n[t,i] - sim.c[t,i] - sim.d[t,i]

            if t< par.T-1:
                sim.p[t+1,:] = sim.psi[t+1,:] * sim.p[t,:]**(par.Lambda)
                sim.y[t+1,:] = sim.zeta[t+1,:] * sim.p[t+1,:]
                sim.m[t+1,:] = par.R * sim.a[t,:] + sim.y[t+1,:]
                sim.n[t+1,:] = (1-par.delta) * sim.d[t,:] 

        toc = process_time()
        print(f'Simulation time: {toc-tic:.2f} seconds')