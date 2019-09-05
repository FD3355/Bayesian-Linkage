# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:34:05 2019

@author: gameg
"""

#from datetime import timedelta,date
#import calendar
#This isn't 1 to 1 with R's date_decmil i.e. see 2018.4567 but for the purposes of our simulated date it works fine
import numpy as np
import math

def GetDate(floatDate, Indvidual_dates = True,array = False):
    if(array==False):
        #floatDate=2018 # For testing
        Year=floatDate
        numstring = str(Year)
        Year = int(numstring[:numstring.find('.')])
        
        Month=floatDate
        numstring = str(Month)
        
        Month = float(numstring[numstring.find('.'):])
        Month = (Month)*12
        
        #Save for days first
        Day=Month
        numstring = str(Day)
        Day = float(numstring[numstring.find('.'):])
    
        #Now we can get month
        Month=str(Month+1)
        Month = int(Month[:Month.find('.')])
        
        
        """
        Months with 31	- January - March - May - July - August - October - December
        Months with 30	- April - June - September - November
        Month with 28*	- February
        *28 days during most years, 29 days during leap years (2000,2004,2008,2012, etc.)
          
        """
        Thirty_One=[1,3,5,7,8,10,12]
        Thirty=[4,6,9,11]
        
      
        if any(Month == a for a in Thirty_One):
              Day=math.ceil(31*Day)
        elif any(Month == a for a in Thirty):
              Day=math.ceil(30*Day)
        elif Month % 4 == 0:
              Day=math.ceil(29*Day)
        else:
              Day=math.ceil(28*Day)
        
        if(Indvidual_dates == True):
            return(Year,Month,Day)
        else:
            return(np.array(Year,Month,Day))
            
    