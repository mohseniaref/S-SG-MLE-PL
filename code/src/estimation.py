
# --------------------------------------------------------------------------------------------------
# import librairies 
# --------------------------------------------------------------------------------------------------
import numpy as np

# --------------------------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------------------------
def SCM(X):  
    """ A function that computes the ML Estimator for covariance matrix estimation for gaussian data
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
        Outputs:
            * sigma_mle = the ML estimate"""
    p = X.shape[0]
    n = X.shape[1] 

    # initialization
    sigma_mle = np.zeros((p, p)) 
  
    sigma_mle = (X@X.conj().T) / n 
    return sigma_mle

def tyler_estimator_covariance(X, tol=0.001, iter_max=100):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimated covariance matrix
            * τ = the estimated tau
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = X.shape
    stop_cond = np.inf # Distance between 2 iterations
    Sigma = np.eye(p) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (stop_cond>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ_est = np.diagonal(X.conj().T@np.linalg.inv(Sigma)@X)
        X_bis = X / np.sqrt(τ_est)
        Sigma_new = (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        stop_cond = np.linalg.norm(Sigma_new - Sigma, 'fro')/ np.linalg.norm(Sigma, 'fro')

        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    return Sigma, iteration

def phasecorrection3(covmatrix): 
    """
    A function that corrects the phase of a covariance matrix to ensure phase alignment with the first element.
        Inputs:
            * covmatrix : Covariance matrix representing the phase information.
        Outputs:
            * phase_minus0 : corrected phases ensuring alignment with the phase of the first element. """
    phase = -np.angle(covmatrix[0,:])
    phase_minus0 = phase-phase[0]
    return phase_minus0

def phasecorrection4(covmatrix,sigmamatrix):
    """    
    A function that corrects the phase of a covariance matrix based on the sub-diagonal elements and sigma matrix.
        Inputs:
            * covmatrix : Covariance matrix 
            * sigmamatrix : Coherence matrix
        Outputs:
        * phase : corrected phases based on sub-diagonal elements and coherence matrix. """
    subdiagphase = np.zeros((sigmamatrix.shape[0]-1))
    phase = np.zeros((sigmamatrix.shape[0]))
    for i in range (sigmamatrix.shape[0]-1):
        subdiagphase[i] = ((np.angle(covmatrix[i,i+1])) +np.pi)%(2*np.pi)-np.pi
        phase[i+1] = (phase[i]+subdiagphase[i]+np.pi)%(2*np.pi)-np.pi
    return -phase

def PhaseLinking(S, iter_max_PL):
    """ A function that computes the complex vector of the phases
        Input : 
            * S : samples
            * iter_max_MM : number of maximum iteration
        Outputs : 
            * w : complex vector """
    p = S.shape[0]
    S = np.asarray(S, dtype=np.complex128)
    M = np.multiply(np.linalg.inv(abs(S.astype("complex128"))), S.astype("complex128"))
    _, D, _ = np.linalg.svd(M) # SVD
    lambda_max = D[0] # the largest eigenvalue
    lambdaI_minus = lambda_max*(np.eye(p)) - M
    w = np.ones((p, 1))
    for i in range(iter_max_PL):
        w_tilde = lambdaI_minus.dot(w)
        w = np.exp(1j*np.angle(w_tilde))
    return w

def MM_PL(Sigma, S, w, iter_max_MM):
    """ Function that represents the MM algorithm for Phase-Linking Problem
        Inputs : 
            * Sigma : real core of the copvariance matrix
            * S : samples 
            * w : vector of the exponential of the phases
            * w_start : starting point of w
        Outputs : 
            * w : vector of the exponential of the phase 
            * diag_w : the diagonal form of w
            * Cov : the covariance matrix"""
    # iteration = 0
    p = Sigma.shape[0]
    A = np.multiply(np.linalg.inv(Sigma),S)
    A=np.asarray(A, dtype = np.complex128)
    _, D, _ = np.linalg.svd(A)
    lambda_max = D[0]
    lambdaI_minus = lambda_max*(np.eye(p)) - A
    for j in range (iter_max_MM):
    # while (iteration < iter_max_MM):
        w_tilde = lambdaI_minus.dot(w) 
        w = np.exp(1j*np.angle(w_tilde))
        diag_w = np.diag(w.squeeze())
        Cov = (diag_w.dot(Sigma)).dot(diag_w.conj().T)
        # iteration = iteration + 1
    return (w, diag_w, Cov)

def MLE_PL(X,model,param_cov,rank,args):
    """
    Parameters
    ----------
        * X : Input dataset with dimension NxL (N: number of acquisition, L: number of px within a resolution cell)
        * model : distribution of dataset- "Gauss" or "ScaledGauss"
        * param_cov : decomposition of covariance matrix - "Mod" or "Cor"
        * rank : truncated rank for Sigma (interger number < N)
        * args : number of iterations for BCD and MM algorithm, respectively

    Returns
    -------
        * w : vector of phase exponentials
        * newphase : vector of phases
        * diag_w : diagonal matrix of the w vector
        * Sigma : cohrence matrix
    """
    (p, n) = X.shape
    (iter_max_BCD, iter_max_MM, phasecorrectionchoice) = args
    scm = SCM(X)
    Cov = scm # initialization of the covariance matrix to a SCM
    Cov = np.asarray(Cov, dtype=np.complex128)
    w = np.ones((p, 1)) # initialization of the w vector to ones
    # w = PhaseLinking(Cov, iter_PL) 
    diag_w = np.diag(w.squeeze())
    for i in range(iter_max_BCD):
        if model == 'Gaussian':
            SCMtilde = scm 
            tau = np.ones((p,1))
        elif model == 'ScaledGaussian':
            tau = np.diagonal(X.conj().T@np.linalg.inv(Cov)@X)/p
            Xnorm = X / np.sqrt(tau)
            SCMtilde = SCM(Xnorm)  # (1/p) * Xnorm@Xnorm.conj().T
        else:
            raise KeyboardInterrupt
            print('Define wrong input model!')
        if param_cov =='Mod-Arg': # classic PL, Cao formulation
            Sigma = abs(((diag_w.conj().T).dot(SCMtilde)).dot(diag_w))
            Sigma = np.asarray(Sigma, dtype = np.float64)
            # low-rank case
            if rank < p:
                u,s,vh = np.linalg.svd(Sigma)
                u_signal = u[:,:rank]
                u_noise = u[:,rank:]
                sigma = np.mean(s[rank:])
                Sigma = u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
        elif  param_cov =='Cor-Arg': # PL formulation based on MLE
            Sigma = (((diag_w.conj().T).dot(SCMtilde)).dot(diag_w)).real
            Sigma = np.asarray(Sigma, dtype = np.float64)
            # low-rank case
            if rank < p:
                u,s,vh = np.linalg.svd(Sigma)
                u_signal = u[:,:rank]
                u_noise = u[:,rank:]
                sigma = np.mean(s[rank:])
                Sigma = u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma*u_noise@u_noise.conj().T

        Sigma = np.asarray(Sigma, dtype = np.float64)
        A = np.multiply(np.linalg.inv(Sigma),SCMtilde)
        A=np.asarray(A, dtype = np.complex128)
        _, D, _ = np.linalg.svd(A)
        lambdamax = D[0]
        lambdaI_minus = lambdamax*(np.eye(p)) - A
        for j in range (iter_max_MM):
            tilde_w = lambdaI_minus.dot(w) 
            w = np.exp(1j*np.angle(tilde_w))
            diag_w = np.diag(w.squeeze())
            Cov = (diag_w.dot(Sigma)).dot(diag_w.conj().T)
    if  phasecorrectionchoice == 3:
        newphase = phasecorrection3(Cov)
        return w, newphase, diag_w, Sigma
    elif phasecorrectionchoice == 4:
        Sigma= (((diag_w.conj().T).dot(SCMtilde)).dot(diag_w)).real
        newphase = phasecorrection4(Cov,Sigma)
        return w, newphase, diag_w, Sigma


def extractMLEfunc(X,model,param_cov,rank,argsMLE):
    vecphase,vecphasediff, diag_w, Sigma  =  MLE_PL(X,model,param_cov,rank,argsMLE)
    vecphasediff = vecphasediff.squeeze()
    vecphase = np.angle(vecphase).squeeze()
    return vecphase,vecphasediff, diag_w, Sigma