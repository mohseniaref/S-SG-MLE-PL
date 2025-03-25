# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import numpy.random as random
import scipy as sc

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def simulateMV(covariance, N,L):
    """ A function that simulate complex gaussian data 
    Inputs : 
        * covariance : the covariance matrix
        * N : integer
        * L : integer
    Outputs : 
        * X : vector of data 
        """
    Csqrt = sc.linalg.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))
    return X

def simulateMVscaledG(covariance, nu, N,L):
    """ A function that simulate complex non gaussian data 
    Inputs : 
        * covariance : the covariance matrix
        * nu : paramter for the gamma distribution
        * N : integer
        * L : integer
    Outputs : 
        * X : vector of gaussian data 
        * Y : vector of scaled gaussian data
        """
    Csqrt = sc.linalg.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))
    tau = random.gamma(nu,1/nu, L ) # size=(L,))
    tau_mat = np.tile(tau,(N,1))
    Y = np.multiply(X,np.sqrt(tau_mat))
    return X, Y

def sampledistributionchoice(trueC,p, N,sampledist):
    """ A function that generate data based on the choice of the distribution (Gaussian or Scaled Gaussian)
    Inputs : 
        * trueC : the true covariance martix
        * size : tuple of the size
        * sample_dist : 'Gaussian' or '(ScaledGaussian, nu)'
    Outputs : 
        * X : data vector"""
    if sampledist == 'Gaussian':
        X = simulateMV(trueC,p,N) # simulate Gaussian distribution
        return X
    elif sampledist[0] == 'ScaledGaussian':
        _,X = simulateMVscaledG(trueC,sampledist[1],p, N) #simulate non-Gaussian distribution
        return X
    
def phasegeneration(choice,maxphase, p):
    """ A function that generate phase differences
    Inputs : 
        * choice : random or linear
        * maxphase : integer
        * N : integer
    Outputs : 
        * delta_thetasim : vector of phase differences """
    if choice == 'random':
        theta_sim = np.array([random.uniform(-np.pi,np.pi) for i in range(p)])
        delta_thetasim0 = np.array((theta_sim-theta_sim[0]))
        delta_thetasim = np.angle(np.exp(1j*delta_thetasim0))
    elif choice  == 'linear':
        delta_thetasim = np.linspace(0,maxphase,p)
    return delta_thetasim

def simulateCov(trueSigma, truetheta):
    """ A function that simulate the true covariance matrix
    Inputs : 
        * trueSigma : the true core of the covariance matrix
        * truetheta : the true phase values
    Outputs : 
        * trueCov : the true covariance matrix """
    diag_theta = np.diag(np.exp(np.dot(1j,truetheta)))
    truecov = (diag_theta.dot(trueSigma).dot(diag_theta.conj().T))
    return truecov