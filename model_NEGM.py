import numpy as np
from types import SimpleNamespace

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
        par.laambda = 1 #Persistency parameter

        # Numerical integration and grids
        # grid settings


        # Shock grid settings
        par.N_perm = 8
        par.N_trans = 8


        