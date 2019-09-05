# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:46:58 2019

@author: Frank DeVone
"""
import numpy as np
import pandas as pd
import scipy.stats
import math
import os
#from datetime import timedelta
#from datetime import datetime as dt
from Time_Function import GetDate #Self created time function
#from LinkFunctions import myprod
from LinkFunctions import LinkingLoopC
from LinkFunctions import GammaPower
import time

#os.chdir('C:/Users/Frank/Desktop/Record linkage')
os.chdir('C:/Users/gameg/Desktop/Record linkage')
#Unfortunatly I could not find a native way in numpy to do this for characteres, thus these functions
def CompareArray(ComparisonArrayA,ComparisonArrayB):
    returnArray=np.zeros_like(ComparisonArrayA, dtype=bool)
    
    for j in range(0,ComparisonArrayA[0].size):
        
        for i in range(0,ComparisonArrayB.size):
            if(ComparisonArrayA[i,j]==ComparisonArrayB[i]):
                returnArray[i,j]=True
            else:
                returnArray[i,j]=False
    return(returnArray)

#This is my first attempt at replicating the R linkage code
#Unlike with the previous files this time I will create the data set in python
############################################################
####Basic Simulation Scenario
############################################################
###There will be two datasets: Data A and Data B.
###Blocking Structure: Data A will have 10 blocks, Data B will have 15 blocks. Each block in Data A will exist in Data B
###Linking Structure: Each block in Data A will have 20 records, each block in Data B will have 30 records
###Each block will have 15 true links 
###N_A = 200; N_B = 450; S = 10; T = 15; N_Mst = 15, N_M = 150

##Linking Variables:
##Age ~ N(40, 2^2) then convert age to DOB
##Gender ~ Bernoulli(0.5)

##Blocking variables:
##Region ~ Multinomial 3 categories
##Status ~ Bernoulli(.8)
##Type ~ Bernoulli(.5)

###The variables MatchID_A and MatchID_B give the true unique identifier for records in file A and file B.
###Similarly, the variables BlockID_A and BlockID_B give the true block identifer in file A and B
############################################################
##########Linking Variables
############################################################
###First simulate linked data

#DOB
V1 = np.random.normal(30,2,450)
V1 = 2018-V1
DOB_year=np.zeros_like(V1)
DOB_day=np.zeros_like(V1)
DOB_month=np.zeros_like(V1)

for i in range(0,V1.size):
    DOB_year[i],DOB_month[i],DOB_day[i]=GetDate(V1[i])

Gender=np.round(np.random.uniform(1,2,450))

#V4: Unique ID number for the linked individuals
MatchID = np.round(np.random.uniform(1000000,10000000,450))

#Merge Linked Data together as a panda data frame
data = {'Match ID': MatchID,'Gender': Gender, 'DOB_year': DOB_year, 'DOB_day': DOB_day, 'DOB_month': DOB_month}
#Working with pandas and np arrays

#numpy is useful for statistical random calculation while pandas is a more visible way to view data

LinkData_A = pd.DataFrame(data)
LinkData_B = pd.DataFrame(data)

############################################################
###Simulate additional unlinked data for Dataset_A
#N_A=200, need to generate data for 50 unlinked individuals

#Distribution of X variables will be the same as the linked data
V1_A = np.random.normal(30,2,150)
V1_A = 2018-V1_A
DOB_year_A=np.zeros_like(V1_A)
DOB_day_A=np.zeros_like(V1_A)
DOB_month_A=np.zeros_like(V1_A)

for i in range(0,V1_A.size): #Not sure if loop is most efficient using fro now
    DOB_year_A[i],DOB_month_A[i],DOB_day_A[i]=GetDate(V1_A[i])

Gender_A=np.round(np.random.uniform(1,2,150))


MatchID_A = np.round(np.random.uniform(1000000,10000000,150))

#I choose to have matchID as a variable instead of an idex as in R
data_A = {'Match ID': MatchID_A,'Gender': Gender_A, 'DOB_year': DOB_year_A, 'DOB_day': DOB_day_A, 'DOB_month': DOB_month_A}
LinkData_A_Unlinked = pd.DataFrame(data_A)
#data_A = pd.DataFrame({'Gender': Gender, 'DOB_year': DOB_year, 'DOB_day': DOB_day, 'DOB_month': DOB_month})

Dataset_A = LinkData_A.append([LinkData_A_Unlinked],sort=False, ignore_index=True)
#pd.merge(LinkData_A,LinkData_A_Unlinked)

############################################################
###Simulate additional unlinked data for N_B
#N_B=450, need to generate data for 300 unlinked individuals

#Distribution of variables will be the same as the linked data
V1_B = np.random.normal(30,2,750)
V1_B = 2018-V1_B
DOB_year_B=np.zeros_like(V1_B)
DOB_day_B=np.zeros_like(V1_B)
DOB_month_B=np.zeros_like(V1_B)

for i in range(0,V1_B.size):
    DOB_year_B[i],DOB_month_B[i],DOB_day_B[i]=GetDate(V1_B[i])

Gender_B=np.round(np.random.uniform(1,2,750))


MatchID_B = np.round(np.random.uniform(1000000,10000000,750))

data_B = {'Match ID': MatchID_B,'Gender': Gender_B, 'DOB_year': DOB_year_B, 'DOB_day': DOB_day_B, 'DOB_month': DOB_month_B}
LinkData_B_Unlinked = pd.DataFrame(data_B)

Dataset_B = LinkData_B.append([LinkData_B_Unlinked],sort=False, ignore_index=True)

############################################################
##########Blocking Variables
############################################################

###Data A

Region_A = np.random.choice(["N","W","MW","S"], size=30, replace=True)
Status_A = np.random.binomial(1,.8,30)
Type_A = np.random.binomial(1,.5,30)
Region_SES_A = np.random.choice(["L","M","H"], size=30, replace=True, p=[.25,.5,.25]) #This is replacing the SES_A variable
Income_A= np.random.normal(50000,10000,30)
Block_A_ID=np.round(np.random.uniform(100,10000,30))
Block_A_S=np.array(range(0,Block_A_ID.size))

###Data B

Region = np.random.choice(["N","W","MW","S"], size=10, replace=True)
Status = np.random.binomial(1,.8,10)
Type = np.random.binomial(1,.5,10)
Region_SES = np.random.choice(["L","M","H"], size=10, replace=True, p=[.25,.5,.25])
Income= np.random.normal(50000,10000,10)

Block_B_ID=np.concatenate([Block_A_ID,np.round(np.random.uniform(100,10000,10))])
Block_B_T=np.array(range(0,Block_B_ID.size))

Region_B=np.concatenate([Region_A,Region])
Status_B=np.concatenate([Status_A,Status])
Type_B=np.concatenate([Type_A,Type])
Region_SES_B=np.concatenate([Region_SES_A,Region_SES])
Income_B=np.concatenate([Income_A,Income])

Dataset_A['Block_ID']=np.concatenate([np.repeat(Block_A_ID,15),np.repeat(Block_A_ID,5)]) 
Dataset_B['Block_ID']=np.concatenate([np.repeat(Block_B_ID,15),np.repeat(Block_B_ID,15)])

Dataset_A['S']=np.concatenate([np.repeat(Block_A_S,15),np.repeat(Block_A_S,5)]) 
Dataset_B['T']=np.concatenate([np.repeat(Block_B_T,15),np.repeat(Block_B_T,15)])

############################################################
###Introduce Error into entries from dataset A
############################################################
#Error in Region
Regionerrorprob=.4
Regionerror=np.random.binomial(1,Regionerrorprob,Region_A.size)

#Define below for use in both loops
Unique=np.unique(Region_A)

for i in range(0,Region_A.size):
  if(Regionerror[i]==1):
      Region_A[i]=np.random.choice(Unique[Unique!= Region_A[i]])
  
#Error in Income
Income_A=Income_A+np.random.normal(0,500/scipy.stats.norm.ppf(1-.4/2),Income_A.size)
#To add in error double check this (the above) first! 
#Income_A=Income_A+np.random.normal(0,500/scipy.stats.norm.ppf(1-.2/2),Income_A.size)
#This just adds 0 same in R

###Error in DOB
DOBerrorprob=.4
DOBerror=np.random.binomial(1,DOBerrorprob,Dataset_A['DOB_month'].size)

Unique=np.unique(Dataset_A['DOB_month'])

for i in range(0,Dataset_A['DOB_month'].size):
  if DOBerror[i]==1:
   Dataset_A['DOB_month'][i]=np.random.choice(Unique[Unique!= Dataset_A['DOB_month'][i]])  
    #N_A$DOB_month[i]=sample(levels(N_A$DOB_month)[levels(N_A$DOB_month)!=N_A$DOB_month[i]],1)

############################################################
##########Linking Variable Comparisons
############################################################ 
    
#Create Comparison matrix Gamma for each of the fields in A and B   
#np.title stacks a new array made up of the input by a given size.  
DOB_year_comparison = np.tile(np.array([Dataset_B.DOB_year]),(Dataset_A.DOB_year.size,1)) 
DOB_month_comparison = np.tile(np.array([Dataset_B.DOB_month]),(Dataset_A.DOB_month.size,1)) 
DOB_day_comparison = np.tile(np.array([Dataset_B.DOB_day]),(Dataset_A.DOB_day.size,1))
Gender_comparison = np.tile(np.array([Dataset_B.Gender]),(Dataset_A.Gender.size,1))

#Compare each variable in dataset A with each element in dataset B element-wise
#DOB_year_gamma_justone=np.equal(DOB_year_comparison[:,0],Dataset_A.DOB_year)
#Above sample code is this done once,below apply_along_axis does it over the entire array
DOB_year_gamma=np.apply_along_axis(np.equal,0,DOB_year_comparison,Dataset_A.DOB_year)
DOB_month_gamma=np.apply_along_axis(np.equal,0,DOB_month_comparison,Dataset_A.DOB_month)
DOB_day_gamma=np.apply_along_axis(np.equal,0,DOB_day_comparison,Dataset_A.DOB_day)
Gender_gamma=np.apply_along_axis(np.equal,0,Gender_comparison,Dataset_A.Gender)


############################################################
##########Blocking Variable Comparisons
############################################################
#Create Comparison matrix Gamma for each of the fields in A and B
Region_comparison = np.tile(np.array([Region_B]),(Region_A.size,1)) 
Status_comparison = np.tile(np.array([Status_B]),(Status_A.size,1)) 
Type_comparison = np.tile(np.array([Type_B]),(Type_A.size,1)) 
SES_comparison = np.tile(np.array([Region_SES_B]),(Region_SES_A.size,1)) 
Income_comparison = np.tile(np.array([Income_B]),(Income_A.size,1)) 

#Compare each variable in dataset A with each element in dataset B element-wise
#np.equal doesn't work for characters, I'm gonna make my own function
'''#Proof the function works
DOB_year_gamma_test=CompareArray(DOB_year_comparison,Dataset_A.DOB_year)
DOB_year_gamma_test==DOB_year_gamma
'''
Region_gamma=CompareArray(Region_comparison,Region_A)
Status_gamma=CompareArray(Status_comparison,Status_A)
Type_gamma=CompareArray(Type_comparison,Type_A)
SES_gamma=CompareArray(SES_comparison,Region_SES_A)
Income_gamma=CompareArray(Income_comparison,Income_A)

############################################################
###Create Gamma matrices from element-wise comparisons
#Date of Birth
#Note:these will throw errors if not converted into float 32s
Gamma2=np.subtract(DOB_year_gamma.astype(np.float32),DOB_month_gamma.astype(np.float32))
Gamma2[Gamma2!=1]=0#This works exactly as in R

Gamma3=np.subtract(np.add(DOB_year_gamma.astype(np.float32),DOB_month_gamma.astype(np.float32)),DOB_day_gamma.astype(np.float32))-1
Gamma3[Gamma3!=1]=0

Gamma4=np.add(np.add(DOB_year_gamma.astype(np.float32),DOB_month_gamma.astype(np.float32)),DOB_day_gamma.astype(np.float32))-2
Gamma4[Gamma4!=1]=0

Gamma1=-np.add(np.add(Gamma2,Gamma3),Gamma4)+1
Gamma1[Gamma1<=0]=0

#For testing, should be all ones, same for 5+6, and pairwise Zetas
#GammaTest=np.add(np.add(np.add(Gamma2,Gamma3),Gamma4),Gamma1) 

#Gender
Gamma6=Gender_gamma
Gamma5=-(Gender_gamma.astype(np.float32))+1
#GammaTest=np.add(Gamma5,Gamma6) 

#Region
Zeta1=-(Region_gamma.astype(np.float32))+1
Zeta2=Region_gamma.astype(np.float32)
#ZetaTest=np.add(Zeta1,Zeta2) 

#For loops in loop
SuperGamma=np.concatenate([Gamma1,Gamma2,Gamma3,Gamma4,Gamma5,Gamma6])

#Status
Zeta3=-(Status_gamma.astype(np.float32))+1
Zeta4=Status_gamma.astype(np.float32)
#ZetaTest=np.add(Zeta3,Zeta4)

#Type
Zeta5=-(Type_gamma.astype(np.float32))+1
Zeta6=Type_gamma.astype(np.float32)
#ZetaTest=np.add(Zeta5,Zeta6)

#SES
Zeta7=-(SES_gamma.astype(np.float32))+1
Zeta8=SES_gamma.astype(np.float32)
#ZetaTest=np.add(Zeta7,Zeta8)

#Income
Zeta9=-(Income_gamma.astype(np.float32))+1
Zeta10=Income_gamma.astype(np.float32)
#ZetaTest=np.add(Zeta9,Zeta10)
start_time = time.time()
#From below it is the same general structure from Python Record Linkage 4.0, but blocking adds a lot of steps
#Specify values for hyperparameters of prior distributions
prior_BM_Region=np.array([1,1]); prior_BM_Status=np.array([1,1]); prior_BM_Type=np.array([1,1]); prior_BM_SES=np.array([1,1]); prior_BM_Income=np.array([1,1])
prior_BU_Region=np.array([1,1]); prior_BU_Status=np.array([1,1]); prior_BU_Type=np.array([1,1]); prior_BU_SES=np.array([1,1]); prior_BU_Income=np.array([1,1])
prior_CM_DOB=np.array([1,1,1,1]); prior_CM_Gender=np.array([1,1])
prior_CU_DOB=np.array([1,1,1,1]); prior_CU_Gender=np.array([1,1])
prior_CBU_DOB=np.array([1,1,1,1]); prior_CBU_Gender=np.array([1,1])
prior_pi=np.array([1,1])

#Set the number of simulations
nsim=125

theta_BM_Region=np.zeros((prior_BM_Region.size,nsim),dtype=float)
theta_BM_Status=np.zeros((prior_BM_Status.size,nsim),dtype=float)
theta_BM_Type=np.zeros((prior_BM_Type.size,nsim),dtype=float)
theta_BM_SES=np.zeros((prior_BM_SES.size,nsim),dtype=float)
theta_BM_Income=np.zeros((prior_BM_Income.size,nsim),dtype=float)

theta_BU_Region=np.zeros((prior_BU_Region.size,nsim),dtype=float)
theta_BU_Status=np.zeros((prior_BU_Status.size,nsim),dtype=float)
theta_BU_Type=np.zeros((prior_BU_Type.size,nsim),dtype=float)
theta_BU_SES=np.zeros((prior_BU_SES.size,nsim),dtype=float)
theta_BU_Income=np.zeros((prior_BU_Income.size,nsim),dtype=float)

theta_CM_DOB=np.zeros((prior_CM_DOB.size,nsim),dtype=float)
theta_CM_Gender=np.zeros((prior_CM_Gender.size,nsim),dtype=float)

theta_CU_DOB=np.zeros((prior_CU_DOB.size,nsim),dtype=float)
theta_CU_Gender=np.zeros((prior_CU_Gender.size,nsim),dtype=float)

theta_CBU_DOB=np.zeros((prior_CBU_DOB.size,nsim),dtype=float)
theta_CBU_Gender=np.zeros((prior_CBU_Gender.size,nsim),dtype=float)

#Initialize Vectors to store links and other linking statistics
BlockDesignation=np.zeros((Block_A_ID.size,nsim),dtype=float)
BlockProbability=np.zeros((Block_A_ID.size,nsim),dtype=float)
BlockMove=np.zeros((Block_A_ID.size,nsim),dtype=float)
BlockMoveType=np.zeros((Block_A_ID.size,nsim),dtype=float)
MoveProbability=np.zeros((Block_A_ID.size,nsim),dtype=float)

LinkDesignation=np.zeros((Dataset_A.shape[0],nsim),dtype=float)
LinkProbability=np.zeros((Dataset_A.shape[0]),dtype=float)

###Initialize blocking, linking, and blocking designation matrices, this is new
Blocking_S = np.transpose(np.tile(np.array([Dataset_A['S']]),(Dataset_B.shape[0],1)))#Check this
Blocking_T = np.tile(np.array([Dataset_B['T']]),(Dataset_A.shape[0],1))#Check this
Blocking_ST = np.zeros((Dataset_A.shape[0],Dataset_B.shape[0]),dtype=float)

Blocking=np.zeros((Block_A_ID.size,Block_B_ID.size),dtype=float)

def block_projection(Blocking,Blocking_S,Blocking_T):   
 ###Write function to project partitions of B onto the linking space B_ST   
    Blocking_ST=np.transpose(np.array(np.where(Blocking==1)))
    #print(Blocking_ST) #For testing
    projection=np.zeros((Blocking_S.shape[0],Blocking_T.shape[1]),dtype=float)
    #if Blocking_ST.size>0: #Ask Ming about this
    for i in range(0,Blocking.shape[0]):
        #print(i) #For Testing
        Block_ind_S=(Blocking_S==Blocking_ST[i,0])
        Block_ind_T=(Blocking_T==Blocking_ST[i,1]) 
        Block_ind=np.multiply(Block_ind_S.astype(np.float32),Block_ind_T.astype(np.float32))
        #Block_ind=myprod(Block_ind_T.astype(np.float32),np.transpose(Block_ind_S.astype(np.float32)))
        #print(np.sum(Block_ind))
        projection=projection+Block_ind
        
    return(projection)
    
###Specify starting values for parameters, blocking, and linking matrix
#Starting value for blocks will be a random permutation of the indexes
BlockDesignation[:,0]=np.random.choice(Block_B_T,size=30,replace=False)

#'''#Fortesting
#BlockDesignation[:,0]=np.array([0,1,2,3,4,5,6,7,8,9])
#'''
###Function to transform block row and column designations into a matrix of 0's and 1's
def create_block_matrix(Block_Designation,B):
  #i=0 #For testing, will send [i] up
  #This seems to work but missing columns, keep for now
  A=pd.DataFrame({'sequence':range(0,BlockDesignation[:,0].size),'BD':BlockDesignation[:,0]})  
  B_mat=np.zeros((B.shape[0],B.shape[1]))
  for i in range(0,A.shape[0]):
    B_mat[A['sequence'][i].astype(np.int),A['BD'][i].astype(np.int)]=1
  return(B_mat)

#Need to transform BlockDesignation to a matrix B
Blocking=create_block_matrix(BlockDesignation[:,0],Blocking)
#print(np.sum(Blocking))

#Project block designations onto B_ST
Blocking_ST=block_projection(Blocking,Blocking_S,Blocking_T)
#print(np.sum(Blocking_ST)) #Should be 6,000

"""#Blocking_ST test
#Test for rows
y=0
for i in range(0,Blocking_ST.shape[0]):
    x=np.sum(Blocking_ST[i,:])
    #print(x)
    if (x==30):
        y=y+1
print(y)  

#Test for columns
y=0
for i in range(0,Blocking_ST.shape[1]):
    x=np.sum(Blocking_ST[:,i])
    #print(x)
    if (x==20):
        y=y+1
print(y)  
"""

#Start with an empty linking matrix
C=np.zeros((Dataset_A.shape[0],Dataset_B.shape[0]))

#Specify starting values for parameters, ask ming where these are coming from
theta_BM_Region[:,0]=np.array([.3,.7])
theta_BU_Region[:,0]=np.array([.66,.34])
theta_BM_Status[:,0]=np.array([.25,.75])
theta_BU_Status[:,0]=np.array([.75,.25])
theta_BM_Type[:,0]=np.array([.4,.6])
theta_BU_Type[:,0]=np.array([.6,.4])
theta_BM_SES[:,0]=np.array([.35,.65])
theta_BU_SES[:,0]=np.array([.65,.35])
theta_BM_Income[:,0]=np.array([.2,.8])
theta_BU_Income[:,0]=np.array([.8,.2])
theta_CM_DOB[:,0]=np.array([.05,.1,.15,.7])
theta_CU_DOB[:,0]=np.array([.7,.15,.1,.05])
theta_CM_Gender[:,0]=np.array([.25,.75])
theta_CU_Gender[:,0]=np.array([.75,.25])
theta_CBU_DOB[:,0]=np.array([.8,.1,.075,.025])
theta_CBU_Gender[:,0]=np.array([.8,.2])

n=25 #For function calls



#This will all eventually be in a loop for(t in 2:nsim) but for now...
for t in range(1,nsim):
    Blocking_ST=block_projection(Blocking,Blocking_S,Blocking_T)
    
    theta_BM_Region[:,t]=np.random.dirichlet(np.array([prior_BM_Region[0]+np.sum(Zeta1*Blocking),\
          prior_BM_Region[1]+np.sum(Zeta2*Blocking)]))
    theta_BU_Region[:,t]=np.random.dirichlet(np.array([prior_BU_Region[0]+np.sum(Zeta1*(1-Blocking)),\
          prior_BU_Region[1]+np.sum(Zeta2*(1-Blocking))]))    
    theta_BM_Status[:,t]=np.random.dirichlet(np.array([prior_BM_Status[0]+np.sum(Zeta3*Blocking),\
          prior_BM_Status[1]+np.sum(Zeta4*Blocking)]))   
    theta_BU_Status[:,t]=np.random.dirichlet(np.array([prior_BU_Status[0]+np.sum(Zeta3*(1-Blocking)),\
          prior_BU_Status[1]+np.sum(Zeta4*(1-Blocking))]))     
    theta_BM_Type[:,t]=np.random.dirichlet(np.array([prior_BM_Type[0]+np.sum(Zeta5*Blocking),\
          prior_BM_Type[1]+np.sum(Zeta6*Blocking)]))     
    theta_BU_Type[:,t]=np.random.dirichlet(np.array([prior_BU_Type[0]+np.sum(Zeta5*(1-Blocking)),\
          prior_BU_Type[1]+np.sum(Zeta6*(1-Blocking))]))      
    theta_BM_SES[:,t]=np.random.dirichlet(np.array([prior_BM_SES[0]+np.sum(Zeta7*Blocking),\
          prior_BM_SES[1]+np.sum(Zeta8*Blocking)]))     
    theta_BU_SES[:,t]=np.random.dirichlet(np.array([prior_BU_SES[0]+np.sum(Zeta7*(1-Blocking)),\
          prior_BU_SES[1]+np.sum(Zeta8*(1-Blocking))]))      
    theta_BM_Income[:,t]=np.random.dirichlet(np.array([prior_BM_Income[0]+np.sum(Zeta9*Blocking),\
          prior_BM_Income[1]+np.sum(Zeta10*Blocking)]))     
    theta_BU_Income[:,t]=np.random.dirichlet(np.array([prior_BU_Income[0]+np.sum(Zeta9*(1-Blocking)),\
          prior_BU_Income[1]+np.sum(Zeta10*(1-Blocking))]))  
        
    theta_CM_DOB[:,t]=np.random.dirichlet(np.array([prior_CM_DOB[0]+np.sum(C*Gamma1*Blocking_ST),\
          prior_CM_DOB[1]+np.sum(C*Gamma2*Blocking_ST),prior_CM_DOB[2]+np.sum(C*Gamma3*Blocking_ST),\
          prior_CM_DOB[3]+np.sum(C*Gamma4*Blocking_ST)]))
    theta_CU_DOB[:,t]=np.random.dirichlet(np.array([prior_CU_DOB[0]+np.sum((1-C)*Gamma1*Blocking_ST),\
          prior_CU_DOB[1]+np.sum((1-C)*Gamma2*Blocking_ST),prior_CU_DOB[2]+np.sum((1-C)*Gamma3*Blocking_ST),\
          prior_CU_DOB[3]+np.sum((1-C)*Gamma4*Blocking_ST)]))
        
    theta_CM_Gender[:,t]=np.random.dirichlet(np.array([prior_CM_Gender[0]+np.sum(C*Gamma5*Blocking_ST),\
          prior_CM_Gender[1]+np.sum(C*Gamma6*Blocking_ST)]))
    theta_CU_Gender[:,t]=np.random.dirichlet(np.array([prior_CU_Gender[0]+np.sum((1-C)*Gamma5*Blocking_ST),\
          prior_CU_Gender[1]+np.sum((1-C)*Gamma6*Blocking_ST)]))
    
    theta_CBU_DOB[:,t]=np.random.dirichlet(np.array([prior_CBU_DOB[0]+np.sum(Gamma1*(1-Blocking_ST)),\
          prior_CBU_DOB[1]+np.sum(Gamma2*(1-Blocking_ST)),prior_CBU_DOB[2]+np.sum(Gamma3*(1-Blocking_ST)),\
          prior_CBU_DOB[3]+np.sum(Gamma4*(1-Blocking_ST))]))
        
    theta_CBU_Gender[:,t]=np.random.dirichlet(np.array([prior_CBU_Gender[0]+np.sum(Gamma5*(1-Blocking_ST)),\
          prior_CBU_Gender[1]+np.sum(Gamma6*(1-Blocking_ST))]))  
    
    #For first loop    
    SuperTheta_M=np.concatenate([theta_CM_DOB,theta_CM_Gender])
    SuperTheta_U=np.concatenate([theta_CU_DOB,theta_CU_Gender])
    
    #For second loop
    SuperTheta_MB=np.concatenate([theta_CBU_DOB,theta_CBU_Gender])
    
    #SuperTheta_M_Block
  
    ########################################################################################################################
      #Within every block, update the linking configurations C_ST(20x30)
      ####Note this can be optimized and performed in parallel, since the blocks are linked independently  
    
    blocking = np.transpose(np.array(np.where(Blocking==1)))
    #This is for checking, would need to change LinkingLoopC to LinkingLoop
    #if t==1:
       # S_ind=(Blocking_S==BlockingOne[0,0])
       # LinkDesignation_ST=np.zeros((np.sum(S_ind[:,0]), nsim),dtype=float)
       # LinkProbability_ST=np.zeros((np.sum(S_ind[:,0]), nsim),dtype=float)         
      
    
    for s in range(0,np.shape(blocking)[0]):
        #Ask Ming, "Do I even need any of these?" 
        S_ind=(Blocking_S==blocking[s,0])
        T_ind=(Blocking_T==blocking[s,1])#Changed to 1 based off r
        ST_ind=(Blocking_S==blocking[s,0])*(Blocking_T==blocking[s,1])
        C_ST_Forloop=np.reshape([C[ST_ind==1]],(np.sum(S_ind[:,0]),np.sum(T_ind[0,:])),order='F')
        
        
        SuperGamma=np.concatenate([Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1]\
                                   ,Gamma4[ST_ind==1],Gamma5[ST_ind==1],Gamma6[ST_ind==1]])
        SuperGamma=np.reshape(SuperGamma,(np.sum(S_ind[:,0])*6,np.sum(T_ind[0,:])))
        #SuperGamma2=np.reshape(SuperGamma,(np.sum(S_ind[:,0])*6,np.sum(T_ind[0,:])),order='F')
        #SuperGamma==SuperGamma2
        
        #Repeat the Gibbs Sampler 10 times to sufficiently sample the blocks, this is the code you wrote begins
          #nsim=25
          #C=C_ST
          #Supergamma=supergamma[ST_ind==1]
          #Theta is exactly the same
        
        #Only C is neccissary
        #FIRST CALL OF FUNCTION#################################################################################################
        C_ST_Forloop=\
        LinkingLoopC(C_ST_Forloop,SuperGamma,SuperTheta_M[:,t],SuperTheta_U[:,t],prior_pi,n)
        C[ST_ind==1]=C_ST_Forloop.flatten()
        C_ST_Flat=C_ST_Forloop.flatten()
        #ENTIRE PROGRAM IS CORRECT BASED UPON FIRST MEETING UP TO THIS POINT
    ########################################################################################################################
    #Propose an update for each block s 
    #print(blocking)
    for s in range(0,np.shape(blocking)[0]):
        #s=0 
        blocking=np.transpose(np.array(np.where(Blocking==1)))
        blocking=blocking[np.sort(np.argsort(blocking[:,0])),:]
        #print(sum(blocking[:,1]==result))
        #Sample a new block designation 'r' for the proposal move. 
        #'result' will contain the set of indeces in Blocking without 't' that 's' is currently partitioned with.
        result=np.random.choice(np.setdiff1d(np.array(range(0,np.shape(Blocking)[0])),blocking[s,1]))
        #####################################################################
    #####There will be two possible moves depending on whether r is partitioned with another block in A or not
    ###Move 1: r is not currently partitioned with any block in A
        if(sum(blocking[:,1]==result)==0):#If this triggers run function with below parameters
          ###First, we need to calculate the potential links for block B_SR
          #Extract the elements that correspond to block SR
          #print("here!")
          '''
          S_ind=(B_S==b[s,1]) #Same
          T_ind=(B_T==b[s,2]) #Becomes T)_ind
          ST_ind=(B_S==b[s,1])*(B_T==b[s,2]) #Diff
          C_ST=matrix(C[ST_ind==1],nrow=apply(S_ind,2,sum)[1],ncol=apply(T_ind,1,sum)[1]) #Just replace T with R
          '''
          S_ind=(Blocking_S==blocking[s,0]) #Same
          R_ind=(Blocking_T==result) #Changed to result
          SR_ind=S_ind*R_ind
          C_SR_Forloop=np.reshape([C[SR_ind==1]],(np.sum(S_ind[:,0]),np.sum(R_ind[0,:])),order='F')
          
          T_ind=(Blocking_T==blocking[s,1])#Changed to 1 based on R
          ST_ind=S_ind*T_ind #Check to make sure this is used or you're missing something
          
          #Need to fix the below so the right Gammas are being used
          SuperGamma=np.concatenate([Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1]\
                                       ,Gamma4[SR_ind==1],Gamma5[SR_ind==1],Gamma6[SR_ind==1]])
          SuperGamma=np.reshape(SuperGamma,(np.sum(S_ind[:,0])*6,np.sum(T_ind[0,:])))
          
          #Now we'll call linking loop
          ###SECOND CALL OF FUNCTION, CHECK WITH MING ABOUT SUPER THETAS, he said to use alt but R appears to not##########################
          C_SR_Forloop=LinkingLoopC(C_SR_Forloop,SuperGamma,SuperTheta_M[:,t],SuperTheta_U[:,t],prior_pi,n)
          C_SR_Flat=C_SR_Forloop.flatten()
          
          ##MH acceptance probability will have 4 parts
          ##Ratio of theta_BM and theta_BU, methods section of Ming's paper
          #print(s)
          #print(t)
          theta_B_num=\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[blocking[s,0],result],Zeta2[blocking[s,0],result]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[blocking[s,0],result],Zeta4[blocking[s,0],result]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[blocking[s,0],result],Zeta6[blocking[s,0],result]))*\
          np.prod(theta_BM_SES[:,t]**np.array(Zeta7[blocking[s,0],result],Zeta8[blocking[s,0],result]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[blocking[s,0],result],Zeta10[blocking[s,0],result]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[blocking[s,0],blocking[s,1]],Zeta2[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[blocking[s,0],blocking[s,1]],Zeta4[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[blocking[s,0],blocking[s,1]],Zeta6[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_SES[:,t]**np.array(Zeta7[blocking[s,0],blocking[s,1]],Zeta8[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[blocking[s,0],blocking[s,1]],Zeta10[blocking[s,0],blocking[s,1]]))
          
          theta_B_den=\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[blocking[s,0],blocking[s,1]],Zeta2[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[blocking[s,0],blocking[s,1]],Zeta4[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[blocking[s,0],blocking[s,1]],Zeta6[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_SES[:,t]**np.array(Zeta7[blocking[s,0],blocking[s,1]],Zeta8[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[blocking[s,0],blocking[s,1]],Zeta10[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[blocking[s,0],result],Zeta2[blocking[s,0],result]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[blocking[s,0],result],Zeta4[blocking[s,0],result]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[blocking[s,0],result],Zeta6[blocking[s,0],result]))*\
          np.prod(theta_BU_SES[:,t]**np.array(Zeta7[blocking[s,0],result],Zeta8[blocking[s,0],result]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[blocking[s,0],result],Zeta10[blocking[s,0],result]))
          
          ##Ratio of theta_CM and theta_CU
          #I made the function GammaPower to condense this
          theta_C_num=GammaPower(theta_CM_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],C_SR_Flat)+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],C=C_SR_Flat,Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],C=1-C_SR_Flat,Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],C=1-C_SR_Flat,Gammas=2)
          #This is about the same as the R output!
          
          theta_C_den=GammaPower(theta_CM_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[ST_ind==1],C=C[ST_ind==1])+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],C=C[ST_ind==1],Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[SR_ind==1],C=1-C[ST_ind==1],Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],C=1-C[ST_ind==1],Gammas=2)
            
          theta_CB_num=GammaPower(theta_CBU_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[ST_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],Gammas=2)
          
          theta_CB_den=GammaPower(theta_CBU_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],Gammas=2)
          
          ###CHECK THESE WITH MING###
          ##Ratio of prior distributions for C
          prior_C_num=math.gamma(np.sum(C_SR_Forloop)+prior_pi[0])*\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])-\
                     np.sum(C_SR_Forloop)+prior_pi[1])/\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])
                     
          prior_C_den=math.gamma(np.sum(C[ST_ind==1])+prior_pi[0])*\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])-\
                     np.sum(C[ST_ind==1])+prior_pi[1])/\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])     

          #MH acceptance probability
          acceptance=min(1,np.prod([theta_B_num/theta_B_den,\
                                    math.exp(theta_C_num-theta_C_den),\
                                    math.exp(theta_CB_num-theta_CB_den),\
                                    prior_C_num/prior_C_den]))
          #print(acceptance)         
          
          if(np.random.uniform(size=1)[0]<acceptance):
                  Blocking[blocking[s,0],result]=1
                  Blocking[blocking[s,0],blocking[s,1]]=0
                  C[ST_ind==1]=0
                  C[SR_ind==1]=C_SR_Flat #Breaks without flatten
                  #print("if move accepted")
                  
          BlockMove[blocking[s,0],t]=result
          BlockMoveType[blocking[s,0],t]=1
          MoveProbability[blocking[s,0],t]=acceptance
          
        else:
            ###First, we need to calculate the potential links for blocks B_SR and B_QT
            #Extract the elements that correspond to block SR and QT
        
          S_ind=(Blocking_S==blocking[s,0]) 
          R_ind=(Blocking_T==result) 
          SR_ind=S_ind*R_ind
          C_SR_Forloop=np.reshape([C[SR_ind==1]],(np.sum(S_ind[:,0]),np.sum(R_ind[0,:])),order='F')
          
          T_ind=(Blocking_T==blocking[s,1]) #Chnaged to 1 based on python, not fixed but less error (FIXED)
          ST_ind=S_ind*T_ind 
          
          qVar=blocking[blocking[:,1]==result]
          qVar=qVar.flat[0] #Need to do it this way for some reason, or you can't grab the first element
          #print(blocking[blocking[:,1]==result])
          #print(qVar)
          Q_ind=(Blocking_S==qVar)
          QT_ind=Q_ind*T_ind
          QR_ind=Q_ind*R_ind
          C_QT_Forloop=np.reshape([C[QT_ind==1]],(np.sum(Q_ind[:,0]),np.sum(T_ind[0,:])),order='F')
          #These appear to be the same as the first call of the function
          SuperGamma=np.concatenate([Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1]\
                                       ,Gamma4[SR_ind==1],Gamma5[SR_ind==1],Gamma6[SR_ind==1]])
          SuperGamma=np.reshape(SuperGamma,(np.sum(S_ind[:,0])*6,np.sum(T_ind[0,:])))
          
        
          ###THIRD CALL OF FUNCTION####################################################################
          C_SR_Forloop=LinkingLoopC(C_SR_Forloop,SuperGamma,SuperTheta_M[:,t],SuperTheta_U[:,t],prior_pi,n)
          C_SR_Flat=C_SR_Forloop.flatten()
          
          ###Estimating linkage in block QT, CALL FUNCTION 4TH TIME#########################################################
          ###Use QT blocking
          SuperGammaQT=np.concatenate([Gamma1[QT_ind==1],Gamma2[QT_ind==1],Gamma3[QT_ind==1]\
                                       ,Gamma4[QT_ind==1],Gamma5[QT_ind==1],Gamma6[QT_ind==1]])
          SuperGammaQT=np.reshape(SuperGamma,(np.sum(S_ind[:,0])*6,np.sum(T_ind[0,:])))
         
          C_QT_Forloop=LinkingLoopC(C_QT_Forloop,SuperGammaQT,SuperTheta_M[:,t],SuperTheta_U[:,t],prior_pi,n)
          C_QT_Flat=C_QT_Forloop.flatten()
          
          ##MH acceptance probability will have 4 parts
          ##Ratio of theta_BM and theta_BU
          #R line 683
          theta_B_num=\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[blocking[s,0],result],Zeta2[blocking[s,0],result]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[blocking[s,0],result],Zeta4[blocking[s,0],result]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[blocking[s,0],result],Zeta6[blocking[s,0],result]))*\
          np.prod(theta_BM_SES[:,t]**np.array(Zeta7[blocking[s,0],result],Zeta8[blocking[s,0],result]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[blocking[s,0],result],Zeta10[blocking[s,0],result]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[blocking[s,0],blocking[s,1]],Zeta2[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[blocking[s,0],blocking[s,1]],Zeta4[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[blocking[s,0],blocking[s,1]],Zeta6[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_SES[:,t]**np.array(Zeta7[blocking[s,0],blocking[s,1]],Zeta8[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[blocking[s,0],blocking[s,1]],Zeta10[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[qVar,blocking[s,1]],Zeta2[qVar,blocking[s,1]]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[qVar,blocking[s,1]],Zeta4[qVar,blocking[s,1]]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[qVar,blocking[s,1]],Zeta6[qVar,blocking[s,1]]))*\
          np.prod(theta_BM_SES[:,t]**np.array(Zeta7[qVar,blocking[s,1]],Zeta8[qVar,blocking[s,1]]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[qVar,blocking[s,1]],Zeta10[qVar,blocking[s,1]]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[qVar,result],Zeta2[qVar,result]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[qVar,result],Zeta4[qVar,result]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[qVar,result],Zeta6[qVar,result]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta7[qVar,result],Zeta8[qVar,result]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[qVar,result],Zeta10[qVar,result]))
          
          #705 Fixed to region, was SES; check with Ming
          
          theta_B_den=\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[blocking[s,0],blocking[s,1]],Zeta2[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[blocking[s,0],blocking[s,1]],Zeta4[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[blocking[s,0],blocking[s,1]],Zeta6[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_SES[:,t]**np.array(Zeta7[blocking[s,0],blocking[s,1]],Zeta8[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[blocking[s,0],blocking[s,1]],Zeta10[blocking[s,0],blocking[s,1]]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[blocking[s,0],result],Zeta2[blocking[s,0],result]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[blocking[s,0],result],Zeta4[blocking[s,0],result]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[blocking[s,0],result],Zeta6[blocking[s,0],result]))*\
          np.prod(theta_BU_SES[:,t]**np.array(Zeta7[blocking[s,0],result],Zeta8[blocking[s,0],result]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[blocking[s,0],result],Zeta10[blocking[s,0],result]))*\
          np.prod(theta_BM_Region[:,t]**np.array(Zeta1[qVar,result],Zeta2[qVar,result]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta3[qVar,result],Zeta4[qVar,result]))*\
          np.prod(theta_BM_Type[:,t]**np.array(Zeta5[qVar,result],Zeta6[qVar,result]))*\
          np.prod(theta_BM_Status[:,t]**np.array(Zeta7[qVar,result],Zeta8[qVar,result]))*\
          np.prod(theta_BM_Income[:,t]**np.array(Zeta9[qVar,result],Zeta10[qVar,result]))*\
          np.prod(theta_BU_Region[:,t]**np.array(Zeta1[qVar,blocking[s,1]],Zeta2[qVar,blocking[s,1]]))*\
          np.prod(theta_BU_Status[:,t]**np.array(Zeta3[qVar,blocking[s,1]],Zeta4[qVar,blocking[s,1]]))*\
          np.prod(theta_BU_Type[:,t]**np.array(Zeta5[qVar,blocking[s,1]],Zeta6[qVar,blocking[s,1]]))*\
          np.prod(theta_BU_SES[:,t]**np.array(Zeta7[qVar,blocking[s,1]],Zeta8[qVar,blocking[s,1]]))*\
          np.prod(theta_BU_Income[:,t]**np.array(Zeta9[qVar,blocking[s,1]],Zeta10[qVar,blocking[s,1]]))
          
          #724 fixed to status
          
          #Exact same as in if half
          theta_C_num_1=GammaPower(theta_CM_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],C_SR_Flat)+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],C=C_SR_Flat,Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],C=1-C_SR_Flat,Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],C=1-C_SR_Flat,Gammas=2)
          
          #Above with Q variables
          theta_C_num_2=GammaPower(theta_CM_DOB[:,t],Gamma1[QT_ind==1],Gamma2[QT_ind==1],Gamma3[QT_ind==1],Gamma4[QT_ind==1],C_QT_Flat)+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[QT_ind==1],Gamma6[QT_ind==1],C=C_QT_Flat,Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[QT_ind==1],Gamma2[QT_ind==1],Gamma3[QT_ind==1],Gamma4[QT_ind==1],C=1-C_QT_Flat,Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[QT_ind==1],Gamma6[QT_ind==1],C=1-C_QT_Flat,Gammas=2)
          
          #Exact same as in if half
          theta_C_den_1=GammaPower(theta_CM_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[ST_ind==1],C=C[ST_ind==1])+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],C=C[ST_ind==1],Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[SR_ind==1],C=1-C[ST_ind==1],Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],C=1-C[ST_ind==1],Gammas=2)
          
          #Above with Q variables
          theta_C_den_2=GammaPower(theta_CM_DOB[:,t],Gamma1[QR_ind==1],Gamma2[QR_ind==1],Gamma3[QR_ind==1],Gamma4[QR_ind==1],C=C[QR_ind==1])+\
          GammaPower(theta_CM_Gender[:,t],Gamma5[QR_ind==1],Gamma6[QR_ind==1],C=C[QR_ind==1],Gammas=2)+\
          GammaPower(theta_CU_DOB[:,t],Gamma1[QR_ind==1],Gamma2[QR_ind==1],Gamma3[QR_ind==1],Gamma4[QR_ind==1],C=1-C[QR_ind==1],Gammas=4)+\
          GammaPower(theta_CU_Gender[:,t],Gamma5[QR_ind==1],Gamma6[QR_ind==1],C=1-C[QR_ind==1],Gammas=2)
          
          ###Ratio of theta_CBM and theta_CBU
          
          #Exact same as in if half
          theta_CB_num_1=GammaPower(theta_CBU_DOB[:,t],Gamma1[ST_ind==1],Gamma2[ST_ind==1],Gamma3[ST_ind==1],Gamma4[ST_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[ST_ind==1],Gamma6[ST_ind==1],Gammas=2)
          
          #Above with Q variables
          theta_CB_num_2=GammaPower(theta_CBU_DOB[:,t],Gamma1[QR_ind==1],Gamma2[QR_ind==1],Gamma3[QR_ind==1],Gamma4[QR_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[QR_ind==1],Gamma6[QR_ind==1],Gammas=2)
          
          #Exact same as in if half
          theta_CB_den_1=GammaPower(theta_CBU_DOB[:,t],Gamma1[SR_ind==1],Gamma2[SR_ind==1],Gamma3[SR_ind==1],Gamma4[SR_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[SR_ind==1],Gamma6[SR_ind==1],Gammas=2)
          
          #Above with Q variables
          theta_CB_den_2=GammaPower(theta_CBU_DOB[:,t],Gamma1[QT_ind==1],Gamma2[QT_ind==1],Gamma3[QT_ind==1],Gamma4[QT_ind==1],Gammas=4)+\
          GammaPower(theta_CBU_Gender[:,t],Gamma5[QT_ind==1],Gamma6[QT_ind==1],Gammas=2)
          
          #Had above as QR needs to be QT
          
          #THROUGH CHECK TO HERE
          
          ##Ratio of prior distributions for C
          prior_C_num=\
          math.gamma(np.sum(C_SR_Forloop)+prior_pi[0])*\
              math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])-\
              np.sum(C_SR_Forloop)+prior_pi[1])*\
          math.gamma(np.sum(C_QT_Forloop)+prior_pi[1])*\
              math.gamma(min(np.sum(Q_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])-
              np.sum(C_QT_Forloop)+prior_pi[1])/\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])*\
          math.gamma(min(np.sum(Q_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])
          
          prior_C_den=\
          math.gamma(np.sum(C[ST_ind==1])+prior_pi[0])*\
              math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])-\
              np.sum(C[ST_ind==1])+prior_pi[1])*\
          math.gamma(np.sum(C[QR_ind==1])+prior_pi[0])*\
              math.gamma(min(np.sum(Q_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])-\
              np.sum(C[QR_ind==1])+prior_pi[1])/\
          math.gamma(min(np.sum(S_ind,axis=0)[0],np.sum(T_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])*\
          math.gamma(min(np.sum(Q_ind,axis=0)[0],np.sum(R_ind,axis=1)[0])+\
                     prior_pi[0]+prior_pi[1])
          
          acceptance=min(1,np.prod([theta_B_num/theta_B_den,\
                                    math.exp(theta_C_num_1-theta_C_den_1),\
                                    math.exp(theta_C_num_2-theta_C_den_2),\
                                    math.exp(theta_CB_num_1-theta_CB_den_1),\
                                    math.exp(theta_CB_num_2-theta_CB_den_2),\
                                    prior_C_num/prior_C_den]))
          #If the proposal move is accepted, update both the linking blocking and linking matrices
          if(np.random.uniform(size=1)[0]<acceptance):
            Blocking[blocking[s,0],result]=1 
            Blocking[blocking[s,0],blocking[s,1]]=0
            Blocking[qVar,blocking[s,1]]=1
            Blocking[qVar,result]=0
            C[ST_ind==1]=0
            C[SR_ind==1]=C_SR_Flat #Ming told me to change this but R has ST 
            C[QR_ind==1]=0
            C[QT_ind==1]=C_QT_Flat
            #print("else move accepted")
            
          BlockMove[blocking[s,0],t]=result
          BlockMoveType[blocking[s,0],t]=2
          MoveProbability[blocking[s,0],t]=acceptance
          #End of else
        #End of for s loop
        
#Export Designations of Linking Matrix after every iteration of updating the linking matrices
    Allowed=np.argwhere(C==1) #Should be no duplicate links
    
    #So if this while hits something is wrong with C!  Just having the function break is uninformative, so this will "save" the run but print a error message 
    ErrorTrack=0
    while(Allowed[np.sort(np.argsort(Allowed[:,0])),:][:,1].size>LinkDesignation[:,t][np.sum(C,axis=1)>=1].size):
        print("Size Error, old links not being properly overwritten") 
        ErrorTrack=ErrorTrack+1
        size=Allowed.shape[0]
        for i in range(1,size):
            if(i>=Allowed.shape[0]):
                #print("here")
                break
            if(Allowed[i,0]==Allowed[i-1,0]):
                Allowed=np.delete(Allowed, (i),axis=0)
                i=i-1
            
    LinkDesignation[:,t][np.sum(C,axis=1)>=1]=Allowed[np.sort(np.argsort(Allowed[:,0])),:][:,1]
    
    #Export Designations of Blocking Matrix after every iteration of the MH within Gibbs sampler
    BlockDesignation[:,t]=blocking[np.sort(np.argsort(blocking[:,0])),:][:,1]
    print(BlockDesignation[:,t])
    print(t)

print("--- %s seconds ---" % (time.time() - start_time))


Properlinks=0
for l in range(0,LinkDesignation.shape[0]):
    if(l==LinkDesignation[l,t]):
        Properlinks=Properlinks+1
    
print("There were",Properlinks,"proper links in this run!")    

LD = pd.DataFrame(LinkDesignation)
LD.to_csv('LinkDesignation_Error_fixedM.csv',index=False)  

BlockDesignationSave = pd.DataFrame(BlockDesignation)
BlockDesignationSave.to_csv('BlockDesignation_Error_fixedM.csv',index=False)  

C = pd.DataFrame(C)
C.to_csv('C_Error_fixedM.csv',index=False)    

B = pd.DataFrame(C)
B.to_csv('B_Error_fixedM.csv',index=False)   
 

#For Testing
#AllowedPrint = pd.DataFrame(Allowed)  
#AllowedPrint.to_csv('Allowed.csv',index=False)   
''' 
#For Multiprocessing https://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python
'''
#500_100 was 2 hours 