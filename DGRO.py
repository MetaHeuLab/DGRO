#Gold Rush Optimizer (GRO)
#Dynamic Gold Rush Optimization: Fusing Worker Adaptation and Salp Navigation for Enhanced Search

import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from scipy.stats import levy


def initialization(SearchAgents_no, dim, lb, ub):
    delta = np.tile(ub-lb, (SearchAgents_no,1))
    lb = np.tile(lb, (SearchAgents_no,1))
    Positions = np.random.rand(SearchAgents_no,dim)* delta + lb
    return Positions


def boundConstraint(newPos, oldPos, lb, ub):
    NP = newPos.shape
    # check the lower bound
    xl = lb #np.tile(lb, (NP, 1))
    pos = newPos < xl
    newPos[pos] = oldPos[pos]
    # check the upper bound
    xu = ub #np.tile(ub, (NP, 1))
    pos = newPos > xu
    newPos[pos] = oldPos[pos]
    return newPos

def checkB(llb, uub, dim, position):
    #lb = np.tile(llb, (dim, 1)).T
    #ub = np.tile(uub, (dim, 1)).T
    ub = uub
    lb = llb
    for jdim in range(dim):
        while position[jdim] > ub[jdim] or position[jdim] < lb[jdim]:
            if position[jdim] > ub[jdim]:
                b = abs(position[jdim] - ub[jdim])
                position[jdim] = ub[jdim] - b

            if position[jdim] < lb[jdim]:
                b = abs(lb[jdim] - position[jdim])
                position[jdim] = lb[jdim] + b

    return position


def SNWA(N, X, dim, Iteration, Max_iteration, fitness, gbest, gbestFitness, lb, ub):
    import math

    c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))
    X = np.transpose(X)
    X_temp = X
    #-----> salp warm start here
    for i in range(N):
        if i < N / 2:
            for j in range(0, dim):
                c2 = np.random.rand() #random.random()
                c3 = np.random.rand() #random.random()
                if c3 < 0.5:
                    X_temp[j, i] = gbest[j] + c1 * (
                        (ub[j] - lb[j]) * c2 + lb[j])
                else:
                    X_temp[j, i] = gbest[j] - c1 * (
                        (ub[j] - lb[j]) * c2 + lb[j])
        elif i>=N/2 and i<N+1:
            #NMRA
            size_b = int(N / 5)
            t1, t2 = np.random.choice(range(size_b, N), 2, replace=False)
            alpha = np.random.uniform()
            for j in range(dim):
                X_temp[j, i]= X_temp[j, i] + np.random.uniform() * (X_temp[j, t1]- X_temp[j, t2])
                
    

    X_temp = np.transpose(X_temp)
    condition = np.logical_and(lb <= X_temp, X_temp<= ub)
    pos_rand = np.random.uniform(lb, ub)
    X_temp =  np.where(condition, X_temp, pos_rand)


    return X_temp



def DGRO(fobj, lb, ub, dim, N, Max_iter):
    lb = np.asarray(lb) * np.ones(dim)
    ub = np.asarray(ub) * np.ones(dim)
    sigma_initial = 2
    sigma_final = 1 / Max_iter
    best_pos = np.zeros(dim)
    best_score = np.inf
    Positions = initialization(N, dim, lb, ub)
    
    Fit = np.full(N, np.inf)
    X_NEW = Positions.copy()
    Fit_NEW = Fit.copy()
    Convergence_curve = np.zeros(Max_iter)
    #Convergence_curve[0] = np.min(Fit)


    iter = 1
    while iter <= Max_iter:
        for i in range(N):
            Fit_NEW[i] = fobj(X_NEW[i,:])
            if Fit_NEW[i] < Fit[i]:
                Fit[i] = Fit_NEW[i]
                Positions[i,:] = X_NEW[i,:]
            if Fit[i] < best_score:
                best_score = Fit[i]
                best_pos = Positions[i,:]
        l2 = ((Max_iter - iter)/(Max_iter-1))**2 * (sigma_initial - sigma_final) + sigma_final
        l1 = ((Max_iter - iter)/(Max_iter-1))**1 * (sigma_initial - sigma_final) + sigma_final
        for i in range(N):
            coworkers = np.random.permutation(N-1)[:2]
            diggers = np.arange(N)
            diggers[i] = -1
            diggers = diggers[diggers >= 0]
            coworkers = diggers[coworkers]
            digger1 = coworkers[0]
            digger2 = coworkers[1]
            m = np.random.rand()
            if m < 1/3:
                for d in range(dim):
                    r1 = np.random.rand()
                    D3 = Positions[digger2,d] - Positions[digger1,d]
                    X_NEW[i,d] = Positions[i,d] + r1 * D3
            elif m < 2/3:
                for d in range(dim):
                    r1 = np.random.rand()
                    A2 = 2*l2*r1 - l2
                    D2 = Positions[i,d] - Positions[digger1,d]
                    X_NEW[i,d] = Positions[digger1,d] + A2*D2
                
            else:
                for d in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    C1 = 2 * r2
                    A1 = 1 + l1 * (r1 - 1/2)
                    D1 = C1 * best_pos[d] - Positions[i,d]
                    X_NEW[i,d] = Positions[i,d] + A1 * D1

            X_NEW[i,:] = checkB(lb, ub, dim, X_NEW[i,:])


        
        Temp_fit = np.full(N, np.inf)
        for i in range(N):
            Temp_fit[i] = fobj(X_NEW[i,:])
        X_NEW = deepcopy(SNWA(N, X_NEW, dim, iter, Max_iter, Temp_fit, best_pos, best_score, lb, ub))
        
       
        Convergence_curve[iter-1] = best_score
        print(best_score)
        iter = iter+1;


    return best_score

