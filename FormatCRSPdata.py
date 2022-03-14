# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Ioannis Ropotos 


"""
Format CRSP data. Create the essential columns:
    date : end of month date in integer format YYYYmmdd
    date_m : end of month date in integer format YYYYmm
    PERMNO : security identifier
    PERMCO : entity/firm identifier
    EXCHCD : code for the market exhcange in which the security is traded
    SHRCD : share code of the security (to subset for orindary common shares)
    CAP : market capitalization at the PERMNO level. It is calculated as 
        PRC*SHROUT
    CAP_W : market cap at the PERMNO level lagged by one month 
    RET : total return of the PERMNO for the current month. Dividends are included
    
The CRSP universe is all securities for the sample 1926-2021 which 
at the time constitutes the entire CRSP monthly tape.
    
    
    

"""


import os
import pandas as pd
import numpy as np


# Main directory
wdir = r'C:\Users\ropot\Desktop\Python Scripts\FamaFrench2015FF5'
os.chdir(wdir)



# -------------------------------------------------------------------------------------
#                FUNCTIONS - START
# ------------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTED MEAN IN A DATAFRAME    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Weighted mean ignoring nan values 
def WeightedMean(x, df, weights):
    """
    Define the weighted mean function
    """
    # Mask both the values and the associated weights
    ma_x = np.ma.MaskedArray(x, mask = np.isnan(x))
    w = df.loc[x.index, weights]
    ma_w = np.ma.MaskedArray(w, mask = np.isnan(w))
    return np.average(ma_x, weights = ma_w)
    
    
  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   MAP DATES TO JUNE DATES       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
        
def JuneScheme(x):
    """
    Use the June-June scheme as in Fama-French.
    
    x must be a datetime object. It returns a June date
    in the integer format of YYYYmm. 
    """
    # Get month and year
    month = x.month
    year = x.year

    # x is mapped to a June date
    if month<=6:
        date_jun = year*100 + 6
    else:
        nyear = year + 1
        date_jun = nyear*100 + 6
            
    return date_jun
            
    
# Function that inputs a dataframe and a date column that applies the June Scheme
# thus creating a new column named 'date_jun'
def ApplyJuneScheme(df, date_col = 'date', date_format = '%Y%m%d'):
    # Isolate the dates in date_col in a separate dataframe    
    dates = pd.DataFrame(df[date_col].drop_duplicates().sort_values(), columns = [date_col])
    # Define the June date column
    dates['date_jun'] = pd.to_datetime(dates[date_col], format = date_format).apply(lambda x: JuneScheme(x)).astype(np.int32)
    # Merge with original dataframe df. 
    # The above process is very efficient since we don't have to deal
    # with all rows of df but only with one set of dates.
    df = pd.merge(df, dates, how = 'left', on = [date_col])
    return df



# -------------------------------------------------------------------------------------
#                FUNCTIONS - END
# ------------------------------------------------------------------------------------




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    IMPORT RAW CRSP DATA       #     
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Import raw CRSP data. \n')



crsp = pd.read_csv(os.path.join(wdir, 'CRSPmonthly1926_OrdinaryShares.txt'), sep = ',')
# Fix column names
crsp.columns = ['permno', 'date'] + [x.strip() for x in crsp.columns[2:]]
# Rename columns 
ccols = {'Ret' : 'RET',
         'Prc' : 'PRC',
         'Shr' : 'SHROUT',
         'EX' : 'EXCHCD',
         'SH' : 'SHRCD',
         'CL' : 'SHRTP'}
crsp = crsp.rename(columns = ccols)
# drop 'permno' as it is the same as 'PERMNO'
crsp.drop(columns = 'permno', inplace = True)


# Dictionary to 32bit type 
ctotype32 =  {'date' :np.int32, 
              'date_m':np.int32, 
              'permno' : np.int32,
              'PERMNO' : np.int32, 
              'PERMCO' : np.int32,
              'RET' : np.float32,  
              'EXCHCD': np.int32, 
              'SHRCD' : np.float32}

# Prices should be positive
crsp['PRC'] = np.abs(crsp['PRC'])
# Define market cap at the PERMNO level
crsp['CAP'] = crsp['PRC']*crsp['SHROUT']
# If market cap is 0, treat it like a null
crsp['CAP'] = np.where(crsp['CAP']==0, np.nan, crsp['CAP'])
# Define one month lagged market cap at the PERmNo level
crsp = crsp.drop_duplicates(subset = ['date', 'PERMNO'], ignore_index = True)
crsp = crsp.sort_values(by = ['PERMNO', 'date']).reset_index(drop = True)
crsp['CAP_W'] = crsp.groupby(['PERMNO'])['CAP'].shift()


# Define date_m
crsp['date_m'] =  (  np.floor(crsp['date']/100)  ).astype(np.int32)
# Define month 
crsp['month'] = (   crsp['date_m'] % 100    ).astype(np.int32)
# Define year
crsp['year'] = (  np.floor(crsp['date']/10000)  ).astype(np.int32)
# Apply June Scheme to CRSP data
crsp = ApplyJuneScheme(crsp, date_col = 'date_m', date_format = '%Y%m')
# Returns that are -66, -77, -88, -99 are mapped to null
crsp['RET'] = np.where(crsp['RET']<-1, np.nan, crsp['RET'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   ISOLATE CRSP RETURNS      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Isolate CRSP returns and save. \n')

# Extract the monthly CRSP returns 
ret_cols = ['date_m', 'PERMNO', 'RET', 'date_jun']
crspm = crsp[ret_cols]
# Save it
crspm.to_csv(os.path.join(wdir, 'CRSPreturn1926m.csv'), index = False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   ISOLATE CRSP CHARACTERISTICS     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Isolate CRSP characteristics and save. \n')

# Isolate the characteristics
char_cols = ['date_m', 'month', 'year', 'PERMNO', 'PERMCO', 'EXCHCD', \
             'SHRCD', 'SHRTP', 'CAP', 'CAP_W', 'date_jun']
crspchars = crsp[char_cols]
# Save it
crspchars.to_csv(os.path.join(wdir, 'CRSPcharacteristics1926m.csv'), index = False)



