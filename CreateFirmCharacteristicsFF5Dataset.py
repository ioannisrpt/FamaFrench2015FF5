# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Ioannis Ropotos

"""

Use CRSP and Compustat data to construct the FirmCharacteristics dataset. This will be 
the master dataset used in constructing all kinds of characteristic sorted portfolios. 


The Compustat data is the whole Compustat universe as downloaded from 
WRDS with only filter being the USD currency.

"""


import os
import pandas as pd
import numpy as np


# Main directory
wdir = r'C:\Users\ropot\Desktop\Python Scripts\FamaFrench2015FF5'
os.chdir(wdir)


# Control execution
do_last_traded = True

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  EXTRA HANDY FOR LINKING DATABASES  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def ForceDatetime(x, force_value, date_format = '%Y%m%d'):
    """
    Function that converts a variable to a datetime object. If the conversion is not 
    possible, force_value is applied.

    """
    try:
        return pd.to_datetime(x, format = date_format)
    except ValueError:
        return force_value
    
  
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




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    CRSP DATA (IMPORT, ISOLATE)       #     
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Import and process CRSP data - START \n')


# Dictionary to type 32bit
cctotype32 =  {'date_m' : np.int32, 
               'date_jun' : np.int32,
               'month' : np.int32, 
               'year' : np.int32,
               'PERMNO' : np.int32,
               'PERMCO' : np.int32,
               'EXCHCD' : np.int32, 
               'SHRCD' : np.int32,
               'CAP' : np.float32, 
               'CAP_W' : np.float32}

# Import the CRSP characteristics
crspchars = pd.read_csv(os.path.join(wdir, 'CRSPcharacteristics1926m.csv')).astype(cctotype32)

ctotype32 = {'date_m' : np.int32,
             'date_jun' : np.int32,
             'PERMNO' : np.float32,
             'RET' : np.float32}

# Import the CRSP monthly returns
crspm = pd.read_csv(os.path.join(wdir, 'CRSPreturn1926m.csv')).astype(ctotype32)



# Keep necessary columns for identification
id_cols = ['date_m', 'month', 'date_jun', 'PERMNO', 'PERMCO', 
           'year', 'EXCHCD', 'SHRCD', 'SHRTP', 'CAP']
crsp = crspchars[id_cols].copy()


# Isolate the december market cap 
cap_dec_cols = ['PERMNO', 'date_m', 'CAP', 'EXCHCD', 'SHRCD', 'date_jun']
cap_dec = crsp.loc[crsp['month'] == 12, cap_dec_cols ].reset_index(drop = True).copy()
# Rename the CAP column
cap_dec = cap_dec.rename(columns = {'CAP': 'CAP_dec'})
print('CAP december has been defined. \n')

# Isolate the june market cap
cap_jun_cols = ['PERMNO', 'date_m', 'CAP', 'date_jun']
cap_jun = crsp.loc[crsp['month'] == 6, cap_jun_cols ].reset_index(drop = True).copy()
print('CAP june has been defined. \n')

del crsp


print('Import and process CRSP data - END \n')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    COMPUSTAT DATA (IMPORT, FORMAT, DEFINE)    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Import, format, define COMPUSTAT data - START \n')

comptotype32 = {'gvkey' : np.int32,
                'datadate' : np.int32,
                'fyear' : np.int32,
                'fyr' : np.int32,          
                'at' : np.float32,         
                'ceq' : np.float32,       
                'cogs' : np.float32,         
                'lt' : np.float32,         
                'mib' : np.float32,         
                'pstk' : np.float32,        
                'pstkl' : np.float32, 
                'pstkrv' : np.float32,   
                'sale' : np.float32,       
                'seq' : np.float32,         
                'txdb' : np.float32,       
                'txditc' : np.float32,      
                'xint' : np.float32,        
                'xsga' : np.float32,                   
                'sic' : np.float32}         

# Import Compustat data 
comp = pd.read_csv(os.path.join(wdir, 'FF5_Compustat_Variables.csv')).astype(comptotype32)
# Define the calendar year from datadate
comp['cyear'] = (np.floor(comp['datadate'] /10000)).astype(np.int32)
comp['cyear_jun'] = comp['cyear'] + 1
# Apply the June scheme to the column 'cyear_jun'.
# The idea is that compustat accounting variables are mapped to June of year t if 
# the fiscal year ended in the calendar year t-1. 
comp = ApplyJuneScheme(comp, date_col = 'cyear_jun', date_format = '%Y')

# ----------------------
# DEFINE BOOK VALUE 'be'
# ----------------------
    
# Definition of book value
def BookValue(seq, ceq, pstk, at, lt, mib, pstkrv, pstkl, txditc, txdb, itcb, fyear):
    """
    Parameters
    ----------
    seq : shareholder equity  
    ceq : common equity 
    pstk : preferred stock 
    at : total assets 
    lt: liabilities 
    mib : minority interest 
    pstkrv : Book value of preferred stock is redemption 
    pstkl : liquidation 
    txditc : deferred taxes and investment tax credit O 
    txdb : deferred taxes and investment tax credit O
    itcb : investment tax credit O
    fyear : fiscal year
    """
    
    # Impute with zeros null values    
    if pd.isnull(at):
        at = 0
    if pd.isnull(lt):
        lt = 0
    if pd.isnull(mib):
        mib = 0
    
    # Define Stockholder's book equity SBE
    if pd.notnull(seq):
        SBE = seq
    else:
        if pd.notnull(ceq):
            if pd.notnull(pstk):
                SBE = ceq - pstk
            else:
                SBE = ceq + at - lt - mib
        else:
            SBE = np.nan
            
    
    # Define book value of preferred stock BVPS
    if pd.notnull(pstkrv):
        BVPS = pstkrv
    else:
        if pd.notnull(pstkl):
            BVPS = pstkl
        else:
            if pd.notnull(pstk):
                BVPS = pstk
            else:
                BVPS = np.nan
                
    # Define deferred taxes DT
    if pd.notnull(txditc):
        DT = txditc
    else:
        DT = txdb + itcb
        
    
    # Check if BVPS is null
    if pd.isnull(BVPS):
        BVPS = 0
    # Check if DT is null
    if pd.isnull(DT):
        DT = 0
    # Check if txdb is null
    if pd.isnull(txdb):
        txdb = 0
        
    # Definition of book value BE.
    # A null value is accepted only from the SBE.
    if fyear< 1993:
        BE = SBE - BVPS + DT - txdb 
    else:
        BE = SBE - BVPS - txdb
        
    if BE< 0 :
        BE = np.nan
    
    return BE
    
  
    
# Column necessary to define book equity 'be'
be_def_cols = ['seq', 'ceq', 'pstk', 'at', 'lt', 'mib', 'pstkrv', 'pstkl', \
               'txditc', 'txdb', 'itcb', 'fyear']
comp['be'] = comp[be_def_cols].apply(lambda x: BookValue(*x), axis = 1)
    

# ----------------------------------
# DEFINE OPERATING PROFITs 'operpro'
# ----------------------------------
    
# Definition of operating profits 'operpro'
def OperProfit(sale, cogs, xsga, xint):
    """
    Parameters
    ----------
    sale : revenue/total sales    
    cogs : cost of goods sold  
    xsga : selling, general, and administrative expenses
    xint : interest expense    
    """
    
    # Check if at least one of cogs, xsga or xint is not null
    one_exists = pd.notnull(cogs) | pd.notnull(xsga) | pd.notnull(xint) 
    # If one_exists is satisfied then imput the nulls with zeros
    if one_exists:
        if pd.notnull(cogs):
            COGS = cogs
        else: 
            COGS = 0
        if pd.notnull(xsga):
            XSGA = xsga
        else:
            XSGA = 0
        if pd.notnull(xint):
            XINT = xint
        else:
            XINT = 0        
        
    
    if pd.notnull(sale) & one_exists:
        operpro = sale - COGS - XSGA - XINT    
    else:
        operpro = np.nan 
        
        
    return operpro


# Column necessary to define operating profits 
operpro_def_cols = ['sale', 'cogs', 'xsga', 'xint']
comp['operpro'] = comp[operpro_def_cols].apply(lambda x: OperProfit(*x), axis = 1)

print('Drop duplicates and sort \n')
# Drop duplicates 
comp = comp.drop_duplicates(subset = ['gvkey', 'date_jun'])
# Sort by GVKEY and date_jun
comp = comp.sort_values(by = ['gvkey', 'date_jun'])


# OPERATING PROFITABILITY
# -----------------------
def getOP(operpro, be):
    if be>0:
        OP = operpro/be
    else:
        OP = np.nan
    return OP


comp['OP'] = comp[['operpro', 'be']].apply(lambda x: getOP(*x), axis = 1)        
        

# INVESTMENT 
# -----------
def getINV(at, at_lag):
    if (at > 0) & (at_lag > 0):
        INV = at/at_lag - 1 
    else:
        INV = np.nan
    return INV

# Define the lagged value of total assets 'at_lag1'
comp['at_lag1'] = comp.groupby('gvkey')['at'].shift(1)
comp['INV'] = comp[['at', 'at_lag1']].apply(lambda x: getINV(*x), axis = 1)

# Rename GVKEY and save
comp = comp.rename(columns = {'gvkey' : 'GVKEY'})
comp.to_csv(os.path.join(wdir, 'comp.csv'), index = False)
    

print('Import, format, define COMPUSTAT data - END \n')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#        MERGE COMPUSTAT AND CRSP DATA      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print('Merge Compustat and CRSP data - START \n')

linktotype32 = {'GVKEY' : np.int32,                        
                'LPERMNO' : np.int32,        
                'LPERMCO' : np.int32,        
                'LINKDT' : np.int32,         
                'naics' : np.float32}       

# Import Compustat/CRSP link table
link_table = pd.read_csv(os.path.join(wdir, 'CompustatLink_Table.csv')).astype(linktotype32)


# Use maximum date if a datetime object could not be returned due to ValueError
force_value = comp['datadate'].max()
# Numerical onversion to replace 'E' with np.nan
link_table['LINKENDDT'] = pd.to_numeric(link_table['LINKENDDT'], errors = 'coerce')
link_table['LINKENDDT'] = link_table['LINKENDDT'].fillna(value = force_value).astype(np.int32)

# Left merge Compustat with Linke table
# ------------------------------------
comp2 = pd.merge(comp, link_table, how = 'left', on = ['GVKEY'])
# Keep only observations where 'datadate' lie between 'LINKDT' and 'LINKENDDT'
mask1 = comp2['LINKDT'] <= comp2['datadate']
mask2 = comp2['datadate']<= comp2['LINKENDDT']
comp2 = comp2[ mask1 & mask2 ].copy() 



 
# Convert to int32
comp2totype32 = {'LPERMNO' : np.int32,        
                'LPERMCO' : np.int32,        
                'LINKDT' : np.int32, 
                'LINKENDDT' : np.int32}
comp2 = comp2.astype(comp2totype32)


# Inner and then left merge Compustat 2 with CRSP data
# -----------------------------------------------------

# At this point I have :
# 26054 LPERMNOs in comp2
# 35355 PERMNOs in cap_dec
# 34439 PERMNOs in cap_jun


# Inner merge with December CRSP data
# After the merge I have 25937 unique PERMNOs in firmchars
firmchars = pd.merge(comp2.rename(columns = {'LPERMNO' : 'PERMNO'}), cap_dec.drop(columns = ['date_m']), how ='inner', on = ['PERMNO', 'date_jun']) 
# Left merge with CRSP june data
firmchars = pd.merge(firmchars, cap_jun, how ='left', on = ['PERMNO', 'date_jun']) 


firmchars = firmchars.drop_duplicates(subset = ['PERMNO', 'date_jun'])
firmchars = firmchars.sort_values(by = ['PERMNO', 'date_jun'])


# -------------------------------
# DEFINE AGGREGATE MARKET CAP 'ME'
# --------------------------------

# The market cap is issued at the PERMNO level but the aggregate cap of the firm
# is needed for the construction of factors. Thus we need to add the market cap
# across all PERMNOs per PERMCO/GVKEY for a given date_jun
# Fuck lambda fuctions are slow...
firmchars['ME'] = firmchars.groupby(by = ['GVKEY', 'date_jun'])['CAP'].transform(lambda x: x.sum(min_count = 1))
# Same for december market equity
firmchars['ME_dec'] = firmchars.groupby(by = ['GVKEY', 'date_jun'])['CAP_dec'].transform(lambda x: x.sum(min_count = 1))
# Define CAP_W at the PERMNO level 
firmchars['CAP_W'] = firmchars.groupby('PERMNO')['CAP'].shift()
# Define lagged cap value at the GVKEY level
firmchars['ME_W'] = firmchars.groupby('GVKEY')['CAP_W'].transform(lambda x: x.sum(min_count = 1))



# -------------------------------
# DEFINE BOOK TO MARKET   'BtM'
# --------------------------------

firmchars['BtM'] = firmchars['be']/firmchars['ME_dec']

print('Merge Compustat and CRSP data - END \n')

# Save final dataset
firmchars.to_csv(os.path.join(wdir, 'FirmCharacteristicsFF5.csv'), index = False)



# Do last traded day/month --> date_jun
# This step is necessary for the PortSort class to work properly.
# It enhances the Compustat/CRSP dataset with the last date_jun that a
# a security was traded. These extra rows serve as name/identity
# placeholders so that the sorting process can include the delisted
# securities. If we omit this step, then our dataset will suffer
# from look-ahead bias; we implicitly exclude from July t - June t+1 
# securities that were delisted in that same period.
if do_last_traded:
    
    print('Augment FirmCharacteristics with last traded month - START \n')

    # Import the CRSP return dataset 
    crspm = pd.read_csv(os.path.join(wdir, 'CRSPreturn1926m.csv')).astype(ctotype32)
    # Isolate last traded month
    crspm = crspm.sort_values(by = ['PERMNO', 'date_m'])
    last_traded_m = crspm.drop_duplicates(subset = ['PERMNO'], keep = 'last').reset_index(drop = True)
    
    # Concat with main dataframe
    firmchars2 = pd.concat([firmchars, last_traded_m], axis = 0)
    # Sort by PERMNO, date_jun and date_m
    firmchars2 = firmchars2.sort_values(by = ['PERMNO', 'date_jun','date_m'], ignore_index = True)
    # Drop duplicates date_m/PERMNO pairs and keep the first observation
    firmchars2 = firmchars2.drop_duplicates(subset = ['PERMNO', 'date_jun'], keep = 'first')
    
    # The null values of the CRSP columns correspond to cases when 
    # the security stopped trading (delisted) and the Compustat columns 
    # have captured the last trading instance of the security in a date_jun date.
    # We need to fill the null values with 'PERMNO', 'EXCHCD', 'SHRCD'
    #cap_null_cols = ['PERMNO', 'EXCHCD', 'SHRCD']
    #firmchars[cap_null_cols] = firmchars[cap_null_cols].fillna(method = 'ffill')
    id_cols = ['EXCHCD', 'SHRCD', 'LPERMCO', 'GVKEY']
    firmchars2[id_cols] = firmchars2.groupby('PERMNO')[id_cols].fillna(method = 'ffill')
    
    # Re-define lagged cap value
    firmchars2['CAP_W'] = firmchars2.groupby('PERMNO')['CAP'].shift()  
    firmchars2['ME_W'] = firmchars2.groupby('GVKEY')['CAP_W'].transform(lambda x: x.sum(min_count = 1))

    
    
    firmchars2.to_csv(os.path.join(wdir, 'FirmCharacteristicsFF5_last_traded.csv'), index = False)
    
    print('Augment FirmCharacteristics with last traded month - END \n')




