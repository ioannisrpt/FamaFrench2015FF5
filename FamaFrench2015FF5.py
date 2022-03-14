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

    27 February 2022 - with 'FirmCharacteristicsFF5_last_traded.csv'
    
    1. SMB : 97.08% correlation
    2. HML : 94.86% correlation 
    3. RMW : 91.97% correlation 
    4. CMA : 96.73% correlation   
    
"""






import os
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from PortSort import PortSort as ps

# Main directory
wdir = r'C:\Users\ropot\Desktop\Python Scripts\FamaFrench2015FF5'
os.chdir(wdir)
# Portfolio directory 
ff_folder = 'FF5_portfolios'
FFDIR = os.path.join(wdir, ff_folder)
if ff_folder not in os.listdir(wdir):
    os.mkdir(FFDIR)





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






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      IMPORT-FORMAT DATA                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------
# CRSP data
# ---------

# Import CRSP data
ctotype32 = {'date_m' : np.int32,
             'date_jun' : np.int32,
             'PERMNO' : np.int32,
             'RET' : np.float32}
# crspm2 = pd.read_csv(os.path.join(wdir, 'CRSPmonthlydata1963FF5.csv' )).astype(ctotype32)
crspm = pd.read_csv(os.path.join(wdir, 'CRSPreturn1926m.csv')).astype(ctotype32).dropna()
# Subset it for dates after June 1963
crspm  = crspm[crspm['date_jun']>=196306].reset_index(drop = True)

# Show the format of crspm
print(crspm.head(15))



# --------------------
# FIRM CHARACTERISTICS
# --------------------

# Import FirmCharacteristicsFF5 table
#firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristicsFF5.csv'))
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristicsFF5_last_traded.csv'))
# Subset it for dates after June 1963
firmchars  = firmchars[firmchars['date_jun']>=196306].reset_index(drop = True)
# Subset for EXCHCD
firmchars = firmchars.dropna(subset = ['EXCHCD'])
firmchars = firmchars[firmchars['EXCHCD'].isin(set([1,2,3]))]
# Define NYSE stocks for constructing breakpoints
nyse1 = firmchars['EXCHCD'] == 1
nyse2 = firmchars['SHRCD'] == 10.0
nyse3 = firmchars['SHRCD'] == 11.0
firmchars['NYSE'] = np.where(nyse1 & ( nyse2 | nyse3), 1, 0)
# Subset for ordinary common shares
shrcd = [10, 11, 12]
firmchars = firmchars[firmchars['SHRCD'].isin(set(shrcd))].copy()


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


# Create the 2 Size portfolios. 
# Notice that sorting on size depends on the firm size ('ME')
# but the return weights depend on market cap at the security-PERMNO
# level ('CAP_W'). This is the most intuitive approach.
portchar.FFPortfolios(ret_data = crspm, ret_time_id = 'date_m', 
                      FFcharacteristics = ['ME'], 
                      FFlagged_periods = [1], 
                      FFn_portfolios = [2],
                      FFquantile_filters = [['NYSE', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'CAP_W')
    
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
                      FFquantile_filters = [['NYSE', 1], ['NYSE', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'CAP_W')

    
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
                      FFquantile_filters = [['NYSE', 1], ['NYSE', 1]], 
                      FFdir = FFDIR,
                      weight_col = 'CAP_W')

    
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
                      FFquantile_filters = [['NYSE', 1], ['NYSE', 1]], 
                      FFdir = FFDIR, 
                      weight_col = 'CAP_W')

    
# Renaming the portfolios as per Fam& French (2015) 
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
cumret = np.log( 1+ ff5_all_c).cumsum()

# SMB
plt.figure()
cumret[['mySMB', 'SMB']].plot()
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend(['my SMB' , 'Original SMB'])
plt.savefig(os.path.join(wdir, 'SMB_comparison.png'))


# HML
plt.figure()
cumret[['myHML', 'HML']].plot()
plt.ylabel('Cumulative return')
plt.xlabel('Date')
plt.legend(['my HML' , 'Original HML'])
plt.savefig(os.path.join(wdir, 'HML_comparison.png'))


# RMW
plt.figure()
cumret[['myRMW', 'RMW']].plot()
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend(['my RMW' , 'Original RMW'])
plt.savefig(os.path.join(wdir, 'RMW_comparison.png'))

# CMA
plt.figure()
cumret[['myCMA', 'CMA']].plot()
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend(['my CMA' , 'Original CMA'])
plt.savefig(os.path.join(wdir, 'CMA_comparison.png'))



