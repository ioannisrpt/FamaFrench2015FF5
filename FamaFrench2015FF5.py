# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Ioannis Ropotos

"""

Last tested at 26 February 2022


Replication of the five Fama-French factors for the sample period July 1973 to July 2020 using
monthly returns. This is a test for the PortSort class and the FFPortfolios function in which
I compare the correlation of my factors to the original. Any correlation below 95% is considered 
a failure from my side.

The paper can be found at 
https://www.sciencedirect.com/science/article/pii/S0304405X14002323


Results:
--------
    1 November 2021 - with 'FirmCharacteristics.csv'
    
    1. SMB : 96.0% correlation
    2. HML : 87.0% correlation
    3. RMW : 84.0% correlation
    4. CMA : 77.8% correlation 
    
    6 February 2022 - with 'FirmCharacteristicsFF5.csv'
    
    1. SMB : 97.2% correlation
    2. HML : 84.7% correlation --> 87.5% with BtM2 (dgtw)
    3. RMW : 91.1% correlation --> 91.7% with OP2 (dgtw)
    4. CMA : 97.2% correlation     
    
    27 February 2022 - with 'FirmCharacteristicsFF5_last_traded.csv'
    
    1. SMB : 97.1% correlation
    2. HML : 90.2% correlation 
    3. RMW : 91.4% correlation 
    4. CMA : 97.2% correlation   
    
   
Notes:
------
    1.There is something not quit right with factors related to Book Equity. Both HML and RMW exhibit 
    low correlations with the original Fama-French factors. 

    2. When merging compustat data (~26000 LPERMNOs) with CRSP data (~25000 PERMNOs), I end up
    with only 21000 PERMNOs. I do not understand why this is happening. The difference is robust 
    to EXCHCD = 1,2,3 and SHRCD  = 10, 11. Why is that? I did the merge of Compustat data with 
    Compustat/CRSP link table correctly (I checked the code of others who end up replicating HML with
    98% correlation).

"""






import os
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from PortSort import PortSort as ps

# Main directory
wdir = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\FamaFrench2015FF5'
os.chdir(wdir)
# Portfolio directory 
FFDIR = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\FamaFrench2015FF5\FFDIR'





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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  DEFINITION OF BOOK VALUE 'be'   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Definition of book value as in Daniel et al. (2020).
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  DEFINITION OF OPERATING PROFITS 'operpro'  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OPERATING PROFITABILITY 'OP  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def getOP(operpro, be):
    if be>0:
        OP = operpro/be
    else:
        OP = np.nan
    return OP

# ~~~~~~~~~~~~~~~~~~~~~~
#     INVESTMENT 'INV  #
# ~~~~~~~~~~~~~~~~~~~~~~


def getINV(at, at_lag):
    if (at > 0) & (at_lag > 0):
        INV = at/at_lag - 1 
    else:
        INV = np.nan
    return INV



# -------------------------------------------------------------------------------------
#                FUNCTIONS - END
# ------------------------------------------------------------------------------------






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      IMPORT-FORMAT DATA                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------
# CRSP data
# ---------

# Import CRSP data
crspm = pd.read_csv(os.path.join(wdir, 'CRSPmonthlydata1963FF5.csv' ))

# Show the format of crspm
print(crspm.head(15))



# --------------------
# FIRM CHARACTERISTICS
# --------------------

# Import FirmCharacteristicsFF5 table
#firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristicsFF5.csv'))
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristicsFF5_last_traded.csv'))

# RE-DEFINE BOOK VALUE 'be2' as in 
# https://www.fredasongdrechsler.com/data-crunching/fama-french

# create preferrerd stock
firmchars['ps']=np.where(firmchars['pstkrv'].isnull(), firmchars['pstkl'], firmchars['pstkrv'])
firmchars['ps']=np.where(firmchars['ps'].isnull(),firmchars['pstk'], firmchars['ps'])
firmchars['ps']=np.where(firmchars['ps'].isnull(),0,firmchars['ps'])
firmchars['txditc']=firmchars['txditc'].fillna(0)

# create book equity
firmchars['be2']=firmchars['seq']+firmchars['txditc']-firmchars['ps']
firmchars['be2']=np.where(firmchars['be']>0, firmchars['be'], np.nan)

# Book to market BtM2
firmchars['BtM2'] = firmchars['be2']/firmchars['ME']
# Operating profatibility OP2
firmchars['OP2'] = firmchars[['operpro', 'be2']].apply(lambda x: getOP(*x), axis = 1)


# DEfine CAP_W at the PERMNO level 
firmchars = firmchars.sort_values(by = ['PERMNO', 'date_jun'], ignore_index = True)
firmchars['CAP_W'] = firmchars.groupby('PERMNO')['CAP'].shift()


# RE-DEFINE BOOK VALUE 'be3' as 'seq' -- Simplest defintion
firmchars['be3'] = np.where(firmchars['seq']>=0, firmchars['seq'], np.nan)
firmchars['BtM3'] = firmchars['be3'] /firmchars['ME']
firmchars['OP3'] = firmchars[['operpro', 'be3']].apply(lambda x: getOP(*x), axis = 1)

print(firmchars.head(20))

# -------------------
# FAMA-FRENCH FACTORS
# -------------------

# Import original 5 Fama-French factors
ff5 = pd.read_csv(os.path.join(wdir, 'FF5_monthly.csv'))


# Show the format of ff5
print(ff5.head(15))



# Define PortSort class for the construction of all portfolios
portchar = ps.PortSort(df = firmchars,
                       entity_id = 'PERMNO',
                       time_id = 'date_jun',
                       save_dir = FFDIR)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        SMB FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2 Size portfolios
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', 
                      FFcharacteristics = ['ME'], 
                      FFlagged_periods = [1], 
                      FFn_portfolios = [2],
                      FFquantile_filters = [['EXCHCD', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'ME_W')
    
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
size_def = {1 : 'S', 2 : 'B'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
size_p = portchar.FFportfolios.copy().rename(columns = size_def)

# Define the SMB factor (simplest form)
size_p['mySMB']  = size_p['S'] - size_p['B']


# Comparison of mySMB and SMB
smb_comp = pd.merge(size_p['mySMB'], ff5.set_index('date_m')['SMB'], how = 'inner', on = 'date_m')
print('--- Comparison of mySMB and Fama-French SMB factor ----  \n')
print(smb_comp.corr())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        HML FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2x3 Size and Book-to-Market portfolios 
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', 
                      FFcharacteristics = ['ME', 'BtM'],
                      FFlagged_periods = [1, 1], 
                      FFn_portfolios = [2, np.array([0, 0.3, 0.7]) ],
                      FFquantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'ME_W')

    
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
# BtM : 1 = Low, 2 = Neutral, 3 = High
sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
               '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizebtm_p = portchar.FFportfolios.copy().rename(columns = sizebtm_def)

# Define the HML factor
sizebtm_p['myHML'] = (1/2)*(sizebtm_p['SH'] + sizebtm_p['BH']) - \
                     (1/2)*(sizebtm_p['SL'] + sizebtm_p['BL'])  
                     
# Comparison of myHML and HML
hml_comp = pd.merge(sizebtm_p['myHML'], ff5.set_index('date_m')['HML'], how = 'inner', on = 'date_m')
print('--- Comparison of myHML and Fama-French HML factor ----  \n')
print(hml_comp.corr())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        RMW FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the 2x3 Size and Profitability portfolios 
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', 
                      FFcharacteristics=  ['ME', 'OP'], 
                      FFlagged_periods = [1,1], 
                      FFn_portfolios = [2, np.array([0, 0.3, 0.7]) ], 
                      FFquantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], 
                      FFdir = FFDIR,
                      weight_col = 'ME_W')

    
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
# OP : 1 = Weak, 2 = Neutral, 3 = Robust
sizermw_def = {'1_1' : 'SW', '1_2' : 'SN', '1_3' : 'SR', \
               '2_1' : 'BW', '2_2' : 'BN', '2_3' : 'BR'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizermw_p = portchar.FFportfolios.copy().rename(columns = sizermw_def)

# Define the RMW factor
sizermw_p['myRMW'] = (1/2)*(sizermw_p['SR'] + sizermw_p['BR']) - \
                     (1/2)*(sizermw_p['SW'] + sizermw_p['BW'])  
                     
# Comparison of myRMW and RMW
rmw_comp = pd.merge(sizermw_p['myRMW'], ff5.set_index('date_m')['RMW'], how = 'inner', on = 'date_m')
print('--- Comparison of myRMW and Fama-French RMW factor ----  \n')
print(rmw_comp.corr())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        CMA FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2x3 Size and Investment portfolios 
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', 
                      FFcharacteristics = ['ME', 'INV'], 
                      FFlagged_periods = [1,1], 
                      FFn_portfolios = [2, np.array([0, 0.3, 0.7]) ], 
                      FFquantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'ME_W')

    
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
# INV : 1 = Conservative, 2 = Neutral, 3 = Aggressive
sizecma_def = {'1_1' : 'SC', '1_2' : 'SN', '1_3' : 'SA', \
               '2_1' : 'BC', '2_2' : 'BN', '2_3' : 'BA'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizecma_p = portchar.FFportfolios.copy().rename(columns = sizecma_def)

# Define the CMA factor
sizecma_p['myCMA'] = (1/2)*(sizecma_p['SC'] + sizecma_p['BC']) - \
                     (1/2)*(sizecma_p['SA'] + sizecma_p['BA'])  
                     
# Comparison of myCMA and CMA
cma_comp = pd.merge(sizecma_p['myCMA'], ff5.set_index('date_m')['CMA'], how = 'inner', on = 'date_m')
print('--- Comparison of myCMA and Fama-French CMA factor ----  \n')
print(cma_comp.corr())




# ----------------
# SAVE ALL FACTORS
# ----------------

factors_ls =[smb_comp, hml_comp, rmw_comp, cma_comp]

ff5_all = reduce(lambda a,b: a.join(b), factors_ls)

# Save it
ff5_all.reset_index().to_csv(os.path.join(wdir, 'myFF5_monthly.csv'), index = False)

# ----------------
# PLOT PERFORMANCE
# ----------------


ff5_all_c = ff5_all.copy()

ff5_all_c.index = pd.to_datetime(ff5_all_c.index , format = '%Y%m')

np.log( 1+ ff5_all_c).cumsum().plot()

