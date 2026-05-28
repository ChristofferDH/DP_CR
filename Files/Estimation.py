import numpy as np
import scipy.optimize as optimize

def simulatedMoments(model, est_par, theta0, data):
    data.moments = calculateMoments(model.par, data)

    objective_function = lambda x: sumOfSquaredDifferences(x,model,est_par,data)
    results = optimize.minimize(objective_function,theta0, method='BFGS')
    
    return results

def calculateMoments(par, data):
    time_grid = np.arange(par.moments_minage,par.moments_maxage+1)-par.age_min+1 # define the cell which correspond to the age we want the mean for. e.g. age 40-55 --> agegrid: 16-31
    return np.mean(data.A[time_grid,:],1)

def sumOfSquaredDifferences(theta, model, est_par, data):

    #Update parameters
    par = model.par
    par = updateParameters(par,est_par,theta)

    # Solve the model
    model.create_grids()
    model.solve()

    # Simulate the momemnts
    moments = np.nan + np.zeros((data.moments.size,par.moments_numsim))
    for s in range(par.moments_numsim):

        # Simulate
        model.simulate()

        #Calculate moments
        moments[:,s] = calculateMoments(par,model.sim)

    # Mean of moments         
    moments = np.mean(moments,1)

    # Objective function
    if hasattr(par, 'weight_mat'):
        weight_mat_inv = np.linalg.inv(par.weight_mat)  
    else:
        weight_mat_inv = np.eye(moments.size)   # The identity matrix and I^-1=I
    
    diff = (moments-data.moments).reshape(moments.size,1)
    
    return (diff.T @ weight_mat_inv @ diff)

def updateParameters(par, par_names, par_values):
    for i,parval in enumerate(par_values):
        parname = par_names[i]
        setattr(par,parname,parval) 
    return par