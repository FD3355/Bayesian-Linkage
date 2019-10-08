# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:42:20 2019

@author: gameg
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:08:55 2019

@author: gameg
"""

import numpy as np
#import os
#os.chdir("C:/Users/gameg/Desktop/Research")

#Allows for proper matrix multiplication with array objects
def myprod(x,y,z=1):
    product=np.multiply(np.transpose(np.multiply(x,np.transpose(y))),z)
    return product

def Tstack(a):
  return(np.transpose(np.stack(a)))    
def TstackGamma(a=1,b=1,c=1,d=1,C=1):
  return(np.transpose(np.stack([a*C,b*C,c*C,d*C]))) 
  
def GammaPower(theta,a=1,b=1,c=1,d=1,C=1,Gammas=4):
  if(Gammas==1):
      return(np.log(np.prod(np.power(theta,np.transpose(np.stack([a*C])))))) 
  if(Gammas==2):
      return(np.log(np.prod(np.power(theta,np.transpose(np.stack([a*C,b*C])))))) 
  if(Gammas==3):
      return(np.log(np.prod(np.power(theta,np.transpose(np.stack([a*C,b*C,c*C])))))) 
  if(Gammas==4):
      return(np.log(np.prod(np.power(theta,np.transpose(np.stack([a*C,b*C,c*C,d*C])))))) 


def LinkingLoop(C,SuperGamma,IterationTheta_M,IterationTheta_U,prior_pi,nsim):
    
    #Define these to fill and return
    LinkProbabilityToReturn=np.zeros(C.shape[0])   
    LinkDesignationToReturn=np.zeros(C.shape[0])
    
    #Having this variable will make the writing of the below more clear
    GammaHeight=C.shape[0]

#Iterate through the rows of i, note that this is currently taking longer in Python
    for i in range(0,C.shape[0]):#Maybe this should be C.shape[0]-1
        
        #Resetting the result for row i
        C[i,]=0
        #print("Successful C at i=",i)
        #Extracting individuals in file B that have links, not including the individual currently linked to i
        B_linked=np.array(range(0,C.shape[1]))[np.sum(C,axis=0)==1]
        
        #If there are no links, create empty link vector. Then extract non-linked individuals in dataset B
        if B_linked.size==0: #This is really only for the first iteration for t, first few for i
          B_unlinked=np.array(range(0,C.shape[1]))
        else:#Throws an error when I try to run it with no links, need to find an example with links
          B_unlinked=np.setdiff1d(np.array(range(0,C.shape[1])), B_linked, assume_unique=True)
          #B_unlinked finds the xth element of np.array(range(0,C.shape[1])) and removes it, these-
          #are thelinks we already have
        #print("Successful B_Linked at i=",i)
        #Okay so you gotta do this all at once insted of breaking it up write a loop to to fill
        #an array and then multiply the element together
        Location=C.shape[0] #Tracks location in Gamma
        Past=0
        GammaPower=np.c_[SuperGamma[:Location,][i,B_unlinked]] #Set up matrix first
        Past=Location
        Location=Location+C.shape[0]
        if SuperGamma.shape[0]/GammaHeight>1:#If we only have one linking variable we stop here
            for q in range(1,SuperGamma.shape[0]//GammaHeight):#//gives an int
                GammaPower=np.c_[GammaPower,SuperGamma[Past:Location,][i,B_unlinked]]
                Past=Location
                Location=Location+C.shape[0]
        #After this we have what we would expect from before with the ind Gammas
        #print("Successful gammapower at i=",i)
        Theta_MbyGamma=IterationTheta_M**GammaPower
        num=Theta_MbyGamma.prod(axis=1) #Confirmed with old code this num is correct
        
        Theta_UbyGamma=IterationTheta_U**GammaPower
        den=Theta_UbyGamma.prod(axis=1) #Confirmed with old code this den is correct
        #Calculate ratio of likelihoods
        #print(den)
        Likelihood=num/den
        
        #Calculate probability of individual i not linking
        p_nolink=(C.shape[1]-B_linked.size)*(C.shape[0]-B_linked.size+prior_pi[1]-1)\
        /(B_linked.size+prior_pi[0])
    
        #Parsing together possible moves and move probability
        #This adds 3000+i to the end of the array of B_unlinked    
        B_unlinked=np.append(B_unlinked,C.shape[1]+i)
        B_prob=np.append(Likelihood,p_nolink)/sum(Likelihood,p_nolink)
    
        from collections import Counter
        Counter = Counter(B_prob)
    
        #Sample new bipartite link for individual i
        link_designation=np.random.choice(B_unlinked,size=1,p=B_prob)
    
        #Store information about the link designation sampled
        LinkProbabilityToReturn[i]=B_prob[B_unlinked==link_designation]
    
        if(link_designation<=C.shape[1]-1):
            C[i,link_designation]=1
            LinkDesignationToReturn[i]=link_designation
            
    return(LinkDesignationToReturn,LinkProbabilityToReturn,C)


def LinkingLoopC(C,SuperGamma,IterationTheta_M,IterationTheta_U,prior_pi,nsim):
    
    #Define these to fill and return
    #LinkProbabilityToReturn=np.zeros(C.shape[0])   
    #LinkDesignationToReturn=np.zeros(C.shape[0])
    
    #Having this variable will make the writing of the below more clear
    GammaHeight=C.shape[0]

#Iterate through the rows of i, note that this is currently taking longer in Python
    for i in range(0,C.shape[0]):#Maybe this should be C.shape[0]-1
        
        #Resetting the result for row i
        C[i,]=0
        #print("Successful C at i=",i)
        #Extracting individuals in file B that have links, not including the individual currently linked to i
        B_linked=np.array(range(0,C.shape[1]))[np.sum(C,axis=0)==1]
        
        #If there are no links, create empty link vector. Then extract non-linked individuals in dataset B
        if B_linked.size==0: #This is really only for the first iteration for t, first few for i
          B_unlinked=np.array(range(0,C.shape[1]))
        else:#Throws an error when I try to run it with no links, need to find an example with links
          B_unlinked=np.setdiff1d(np.array(range(0,C.shape[1])), B_linked, assume_unique=True)
          #B_unlinked finds the xth element of np.array(range(0,C.shape[1])) and removes it, these-
          #are thelinks we already have
        #print("Successful B_Linked at i=",i)
        #Okay so you gotta do this all at once insted of breaking it up write a loop to to fill
        #an array and then multiply the element together
        Location=C.shape[0] #Tracks location in Gamma
        Past=0
        GammaPower=np.c_[SuperGamma[:Location,][i,B_unlinked]] #Set up matrix first
        Past=Location
        Location=Location+C.shape[0]
        if SuperGamma.shape[0]/GammaHeight>1:#If we only have one linking variable we stop here
            for q in range(1,SuperGamma.shape[0]//GammaHeight):#//gives an int
                GammaPower=np.c_[GammaPower,SuperGamma[Past:Location,][i,B_unlinked]]
                Past=Location
                Location=Location+C.shape[0]
        #After this we have what we would expect from before with the ind Gammas
        #print("Successful gammapower at i=",i)
        Theta_MbyGamma=IterationTheta_M**GammaPower
        num=Theta_MbyGamma.prod(axis=1) #Confirmed with old code this num is correct
        
        Theta_UbyGamma=IterationTheta_U**GammaPower
        den=Theta_UbyGamma.prod(axis=1) #Confirmed with old code this den is correct
        #Calculate ratio of likelihoods
        #print(den)
        Likelihood=num/den
        
        #Calculate probability of individual i not linking
        p_nolink=(C.shape[1]-B_linked.size)*(C.shape[0]-B_linked.size+prior_pi[1]-1)\
        /(B_linked.size+prior_pi[0])
    
        #Parsing together possible moves and move probability
        #This adds 3000+i to the end of the array of B_unlinked    
        B_unlinked=np.append(B_unlinked,C.shape[1]+i)
        B_prob=np.append(Likelihood,p_nolink)/sum(Likelihood,p_nolink)
    
        from collections import Counter
        Counter = Counter(B_prob)
    
        #Sample new bipartite link for individual i
        link_designation=np.random.choice(B_unlinked,size=1,p=B_prob)
    
        #Store information about the link designation sampled
        #LinkProbabilityToReturn[i]=B_prob[B_unlinked==link_designation]
    
        if(link_designation<=C.shape[1]-1):
            C[i,link_designation]=1
            #LinkDesignationToReturn[i]=link_designation
            
    return(C)
