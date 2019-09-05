# Gamma EM in Python
"""
Created on Fri Jan 18 13:21:27 2019

@author: Frank DeVone
"""

#import pandas as pd 
#Import needed packages
import numpy as np
import os
import math

#Set working directory
#os.chdir("C:/Users/gameg/Desktop/Research")

from numpy import genfromtxt
Gamma= genfromtxt('GammaEM.csv', delimiter=',',skip_header=1,dtype='float')
Gamma=Gamma[:,1:]


#t=0 step begins
nsim=100
#Create blank matricies filled with 0s
theta_M=np.zeros_like(Gamma[:nsim],dtype=float)
theta_U=np.zeros_like(Gamma[:nsim],dtype=float)
pi_M=np.zeros_like(Gamma[:nsim,0],dtype=float)

#######Initialize starting values for parameters
theta_M[0]=np.random.uniform(low=.5, high=1.0, size=Gamma.shape[1])
theta_U[0]=np.random.uniform(low=0, high=.5, size=Gamma.shape[1])
pi_M[0]=np.random.uniform(low=0, high=1, size=1)

#Allows for proper matrix multiplication with array objects
def myprod(x,y,z=1):
    product=np.multiply(np.transpose(np.multiply(x,np.transpose(y))),z)
    return product

#Initialize the log likelihood
loglik=np.zeros((nsim, 1),dtype=float)

for t in range(1,nsim-1):
    product1=(np.power(theta_M[t-1,],Gamma))*(np.power((1-theta_M[[t-1,]]),(1-Gamma)))
    product2=(np.power(theta_U[t-1,],Gamma))*(np.power((1-theta_U[[t-1,]]),(1-Gamma)))

    C=(pi_M[t-1]*product1.prod(axis=1))/\
    ((pi_M[t-1]*product1.prod(axis=1))+\
    ((1-(pi_M[t-1]))*product2.prod(axis=1)))    
    pi_M[t]=np.sum(C)/C.size 
    theta_M[t,]=sum(myprod(C,Gamma))/np.sum(C)
    theta_U[t,]=sum(myprod(1-C,Gamma))/np.sum(1-C)
     
    loglik[t]=sum(myprod(C,math.log(np.abs(pi_M[t])))\
    +np.sum(myprod(C,Gamma,np.log(np.abs(theta_M[t,]))),axis=1)\
    +np.sum(myprod(C,(1-Gamma),np.log(np.abs(1-theta_M[t,]))),axis=1)\
    +myprod((1-C),math.log(np.abs(1-pi_M[t])))\
    +np.sum(myprod((1-C),Gamma,np.log(np.abs(theta_U[t,]))),axis=1)\
    +np.sum(myprod((1-C),(1-Gamma),np.log(np.abs(1-theta_U[t,]))),axis=1))
    print(loglik[t])
   
#Only run if you want to perserve files
saveLog=loglik
np.savetxt("Log 1.0.csv", saveLog)
saveTheta_M=theta_M
np.savetxt("Theta_M 1.0.csv", saveLog)
saveTheta=theta_U
np.savetxt("Theta_U 1.0.csv", saveLog)
savePi=pi_M
np.savetxt("Pi 1.0.csv", saveLog)