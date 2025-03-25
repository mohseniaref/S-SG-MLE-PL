# --------------------------------------------------------------------------------------------------
# import librairies 
# --------------------------------------------------------------------------------------------------
import numpy as np
import warnings

from .utility import add_one_obs, ToeplitzMatrix
from .estimation import (
    phasecorrection3,
    phasecorrection4,
    SCM)

# --------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------

def S_SG_MLE_PL_BCD(X, X_past, x_newdata, C, diag_w_past, iter_max_BCD, phasecorrectionchoice, tol = 0.001):
    # initialization
    stop_cond = np.inf
    iteration = 0

    (p, n) = X_past.shape
    inv_C = np.linalg.inv(C)
    S = SCM(X) # Initialise estimate to SCM
    S = np.asarray(S, dtype=np.complex128)
    w_past = np.diag(diag_w_past)
    
    new_w_theta = np.array(1)
    Sigma = ToeplitzMatrix(0.5, p+1)
    new_past_coherence = Sigma[p, 0:p]
    variance_newdata = Sigma[p, p]

    x_newdata_reshaped = x_newdata.reshape((1, x_newdata.shape[0]))


    while (stop_cond>tol) and (iteration<iter_max_BCD):
        if iteration == iter_max_BCD:
            warnings.warn('Recursive algorithm did not converge')

        # Bloc 1 : Computation of tau_est
        temp11 = new_w_theta.conj()*new_past_coherence@diag_w_past.T@inv_C.conj()@X_past.conj()
        temp12 = new_w_theta*new_past_coherence@diag_w_past.conj().T@inv_C@X_past
        temp13 = (x_newdata_reshaped.conj() - temp11) * (x_newdata_reshaped - temp12)
        temp14 = variance_newdata - new_past_coherence@diag_w_past.conj().T@inv_C@diag_w_past@new_past_coherence.T
        temp15 = np.diagonal(X_past.conj().T@inv_C@X_past)
        tau_est = temp13 / ((p+1) * temp14) + temp15 / (p+1) 
        # Update X
        X_bis = X / np.sqrt(tau_est) 
        SCM_bis = SCM(X_bis)
        X_past_bis = X_bis[0:p, :]
        x_newdata_reshaped_bis = x_newdata_reshaped / np.sqrt(tau_est) 

        # Bloc 2 : Computation of the coherence between the past images and the new one
        temp21 = x_newdata_reshaped_bis@X_past_bis.conj().T@inv_C.conj().T@diag_w_past
        temp22 = x_newdata_reshaped_bis.conj()@X_past_bis.T@inv_C.T@diag_w_past.conj()
        temp23 = new_w_theta.conj()*temp21 + new_w_theta*temp22
        temp24 = diag_w_past.T@inv_C.conj()@X_past_bis.conj()@X_past_bis.T@inv_C.T@diag_w_past.conj()
        temp25 = diag_w_past.conj().T@inv_C@X_past_bis@X_past_bis.conj().T@inv_C.conj().T@diag_w_past
        new_past_coherence = np.sum(temp23, axis = 0) @ np.linalg.inv(temp24 + temp25)
        
        # Bloc 3 : Computation of the phase of the new image
        temp31 = x_newdata_reshaped_bis.conj()@X_past_bis.T @ inv_C.T @ diag_w_past.conj() @ new_past_coherence.T
        temp32 = new_past_coherence @ diag_w_past.T @ inv_C.conj() @ X_past_bis.conj()@X_past_bis.T @ inv_C.T @ diag_w_past.conj() @ new_past_coherence.T
        new_w_theta = (temp31 * (1/temp32)).conj()
        new_w_theta_norm = np.exp(1j * np.angle(new_w_theta)) # projection of new w_theta on the complex circle (radius  = 1)

        # Bloc 4 : Computation of the variance of the new image
        temp41 = new_past_coherence@diag_w_past.T@inv_C.conj()@X_past_bis.conj()
        temp42 = x_newdata_reshaped_bis.conj() - new_w_theta_norm.conj() * temp41
        temp43 = new_past_coherence@diag_w_past.conj().T@inv_C@X_past_bis
        temp44 = x_newdata_reshaped_bis - new_w_theta_norm * temp43
        temp45 = new_past_coherence@diag_w_past.conj().T@inv_C@diag_w_past@new_past_coherence.T
        variance_newdata_new = 1/n * np.sum(temp42 * temp44) + temp45
        
        # stop condition
        stop_cond = np.linalg.norm(variance_newdata_new - variance_newdata) / np.linalg.norm(variance_newdata)

        iteration = iteration + 1
        variance_newdata = variance_newdata_new
        new_w_theta = new_w_theta_norm


    line_vector = new_w_theta*new_past_coherence@diag_w_past.conj().T
    C_tilde = add_one_obs(C, line_vector, variance_newdata)
    C_tilde = np.asarray(C_tilde, dtype=np.complex128)
    w_all = np.append(w_past, new_w_theta)
    diag_w = np.diag(w_all)

    if phasecorrectionchoice == 3:
        new_phase = phasecorrection3(C_tilde)
        new_theta = new_phase[-1]
        return C_tilde, new_theta
    elif phasecorrectionchoice == 4:
        Sigma = (((diag_w.conj().T).dot(SCM_bis)).dot(diag_w)).real
        new_phase = phasecorrection4(C_tilde, Sigma)
        new_theta = new_phase[-1]
        return C_tilde, new_theta

