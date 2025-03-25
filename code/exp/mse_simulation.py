# -------------------------------------------------------------
# Packages
# -------------------------------------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.generation import phasegeneration, simulateCov
from src.utility import ToeplitzMatrix, calculateMSE

from simulation import parallel_Monte_Carlo

import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("MSE simulation of the phase difference")
    parser.add_argument("--l", 
                        type=int, 
                        default=20,
                        help="Number of time stamps in the time series")
    parser.add_argument("--rho", 
                        type=float, 
                        default=0.7, 
                        help="Correlation coefficient for Toeplitz coherence matrix")
    parser.add_argument("--n_list", 
                        type=str, 
                        default=", ".join([str(x) for x in range(20, 400, 20)]),
                        help="List of the size of patch to use")
    parser.add_argument("--n_trials", 
                        type=int, 
                        default=1000, 
                        help="Number of Monte-Carlo Trials")
    parser.add_argument("--phasecorrectionchoice",
                        type=int, 
                        default=4, 
                        help="Choice of the phase correction")
    parser.add_argument("--tol", 
                        type=float, 
                        default=0.001, 
                        help="Treshold for stopping the BCD algorithm")
    parser.add_argument("--sampledist", 
                        type=str, 
                        default="ScaledGaussian", 
                        help="The sample distribution")
    parser.add_argument("--model", 
                        type=str, 
                        default="ScaledGaussian", 
                        help="The model")
    parser.add_argument("--nu",
                        type=int,
                        default=0.1,
                        help="Scaled Gaussian distribution parameter")
    parser.add_argument("--p", 
                        type=int, 
                        default=19, 
                        help="The size of the past time series")
    parser.add_argument("--rank", 
                        type=int, 
                        default=20, 
                        help="The rank of the covariance matrix")
    parser.add_argument("--iter_max_BCD", 
                        type=int, 
                        default=30, 
                        help="Number of iterations of the BCD algorithms")
    parser.add_argument("--iter_PL", 
                        type=int, 
                        default=100, 
                        help="The number of iteration of the PL algorithm")
    parser.add_argument("--iter_max_MM", 
                        type=int, 
                        default=50, 
                        help="The number of iteration of the MM algorithm")
    parser.add_argument("--Multi",
                        type=bool,
                        default=True,
                        help="Parallel computing choice")
    parser.add_argument("--n_threads", 
                        type=int, 
                        default=-1, 
                        help="The number of threads")
    parser.add_argument("--maxphase", 
                        type=int, 
                        default=2, 
                        help="The maximum value of the phase")
    parser.add_argument("--phasechoice", 
                        type=str, 
                        default="linear", 
                        help="Choice of phase: 'linear', or 'random'")

    # Safely parse known arguments and ignore unknown arguments passed by Jupyter
    args, unknown = parser.parse_known_args()

    # Convert `n_list` to a list of integers
    args.n_list = [int(x) for x in args.n_list.split(",")]

    # add `argsMLEPL`
    args.argsMLEPL = [args.iter_PL, args.iter_max_BCD, args.iter_max_MM, args.phasecorrectionchoice]

    args.dist = args.sampledist, args.nu

    # Optional: Print parsed arguments to verify
    print("Parsed arguments:", args)

    print("MSE over size of patch simulation with parameters:")
    for key, val in vars(args).items():
        print(f"  * {key}: {val}")
    
    # phases vector simulation
    true_delta = phasegeneration(args.phasechoice, args.maxphase, args.l) # generate phase with either random or linear. for linear, define last phase is needed
    # Coherence matrix simulation
    SigmaTrue = ToeplitzMatrix(args.rho, args.l)
    # Covariance matrix simulation
    trueCov= simulateCov(SigmaTrue, true_delta)


    new_phase_sequential = [[] for i in range(len(args.n_list))] 
    delta_phases_MLEPL = [[] for i in range(len(args.n_list))] 
    delta_phases_classic_PL = [[] for i in range(len(args.n_list))] 
    delta_2p_insar = [[] for i in range(len(args.n_list))] 
    C_tilde_sequential = [[] for i in range(len(args.n_list))] 

    for key, value in enumerate(args.n_list):
        delta_phases_MLEPL[key], delta_phases_classic_PL[key] , delta_2p_insar[key], new_phase_sequential[key], C_tilde_sequential[key] = parallel_Monte_Carlo(args.p, 
                                                                                                                                                                value, 
                                                                                                                                                                trueCov, 
                                                                                                                                                                args.dist, 
                                                                                                                                                                args.rank, 
                                                                                                                                                                args.argsMLEPL, 
                                                                                                                                                                args.n_trials, 
                                                                                                                                                                args.n_threads, 
                                                                                                                                                                args.tol, 
                                                                                                                                                                args.Multi)
    # MSE computation for the last date for different approaches
    MSE_delta_phases_MLEPL = calculateMSE(np.array(delta_phases_MLEPL)[:, :, args.p], 
                                          true_delta[-1], 
                                          args.n_trials, 
                                          args.n_list)
    MSE_delta_phases_classic_PL = calculateMSE(np.array(delta_phases_classic_PL)[:, :, args.p], 
                                              true_delta[-1], 
                                              args.n_trials, 
                                              args.n_list)
    MSE_delta_2p_insar = calculateMSE(np.array(delta_2p_insar)[:, :, args.p], 
                                      true_delta[-1], 
                                      args.n_trials, 
                                      args.n_list)
    MSE_delta_phases_sequential= calculateMSE(new_phase_sequential, 
                                              true_delta[-1], 
                                              args.n_trials, 
                                              args.n_list)

    # MSE outputs
    plt.figure()
    plt.xlabel('n')
    plt.ylabel('MSE')
    plt.plot(args.n_list, 
             MSE_delta_phases_sequential,
             'o-', 
             color ='blue', 
             label = 'S-SG-MLE-PL')
    plt.plot(args.n_list, 
             MSE_delta_phases_MLEPL,
             'o-', 
             color ='g', 
             label = 'SG-MLE-PL')
    plt.plot(args.n_list, 
             MSE_delta_phases_classic_PL,
             'o-', 
             color ='red', 
             label = 'classic PL')
    plt.plot(args.n_list, 
             MSE_delta_2p_insar,
             'o-', 
             color ='pink', 
             label = '2p-InSAR')
    plt.legend()
    plt.grid("True")
    plt.title('At date '+str(args.l)+', Scaled Gaussian model, p+1='+str(args.p+1)+', rho='+str(args.rho))    
    plt.show()
    
    # histogram outputs for n_samples[3]
    plt.hist(np.array(delta_phases_MLEPL)[3,:,-1],
             bins=np.linspace(-np.pi,np.pi,70),
             label='SG-MLE-PL',
             histtype='step', 
             edgecolor='magenta',  
             fill= True, 
             alpha = 0.25,
             color = 'green' )
    plt.hist(np.array(delta_phases_classic_PL)[3,:,-1],
             bins=np.linspace(-np.pi,np.pi,70),
             label='classic PL',
             histtype='step', 
             edgecolor='b',  
             fill= True, 
             alpha = 0.25,
             color = 'red' )
    plt.hist(np.array(delta_2p_insar)[3,:,-1],
             bins=np.linspace(-np.pi,np.pi,70),
             label='2-p InSAR',
             histtype='step', 
             edgecolor='g',
             lw=1,  
             fill= True, 
             alpha = 0.35,
             color = 'pink' )
    plt.hist(np.array(new_phase_sequential)[3, :],
             bins=np.linspace(-np.pi,np.pi,70),
             label='S-SG-MLE-PL',
             histtype='step', 
             edgecolor='red',
             lw=1,fill= True, 
             alpha = 0.2,
             color = 'blue')
    plt.axvline(x=true_delta[-1],
                color='k', 
                linestyle='-',
                label='simulated phase')
    plt.title('Histogram of phase difference at image '+str(args.l)+' with p='+str(args.p)+', n='+str(args.n_list[3])+' and rho='+str(args.rho))
    plt.legend()
    plt.xlabel('phase (in radian)')
    plt.show()

