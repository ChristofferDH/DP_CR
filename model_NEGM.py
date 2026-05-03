import numpy as np
from types import SimpleNamespace
import Tools

class Durable_BufferStock():
    
    def __init__(self, name = None):
        self.par = SimpleNamespace()
        self.solution = SimpleNamespace()
        self.simulation = SimpleNamespace()

        
    def modelSetup(self):
        par = self.par

        # All of these parameters are taken from Druedahl (2020)
        par.T = 50
        par.beta = 0.965
        par.rho = 2
        par.alpha = 0.9
        par.d_floor = 0.01
        par.R = 1.03
        par.tau = 0.1
        par.delta = 0.15
        par.sigma_perm = 0.1 
        par.sigma_trans = 0.1
        par.Lambda = 1 #Persistency parameter

        # Grids!!!
        # pre and post-decisision state grid
        ### p: grid settings
        par.p_min = 0.0001
        par.p_max = 3
        par.p_N = 150 
        
        ## Post-decision state grids

        ### d: grid settings

        ### a: grid settings
        par.a_min = 0
        par.a_max = 11
        par.a_N = 300 

        ## Pre-decision state grids
        ### x: grid settings
        par.x_min = 0
        par.x_max = 13
        par.x_N = 150 

        ### n: grid settings
        par.n_min = 0
        par.n_max = 3
        par.n_N = 150 

        ### m: grid settings
        par.m_min = 0
        par.m_max = 10
        par.m_N = 300 
        
        # Numerical integration
        ## Shock grid settings
        par.N_perm = 8
        par.N_trans = 8

    def create_grids(self):
        par = self.par

        # State grids

        par.grid_a = Tools.gridFunction(par.a_min,par.a_max, par.a_N)
        par.grid_x = Tools.gridFunction(par.x_min,par.x_max, par.x_N)
        par.grid_n = Tools.gridFunction(par.n_min,par.n_max, par.n_N)
        par.grid_m = Tools.gridFunction(par.m_min,par.m_max, par.m_N)
        par.grid_p = Tools.gridFunction(par.p_min,par.p_max, par.p_N)

        # Quadrature: nodes and weights