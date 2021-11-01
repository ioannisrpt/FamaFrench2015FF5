# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Ioannis Ropotos

"""

Last tested at 1 November 2021


Replication of the five Fama-French factors for the sample period July 1973 to July 2020 using
monthly returns. This is a test for the PortSort class and the FFPortfolios function in which
I compare the correlation of my factors to the original. Any correlation below 95% is considered 
a failure from my side.

The paper can be found at 
https://www.sciencedirect.com/science/article/pii/S0304405X14002323


Results:
--------
    1. SMB : 96% correlation
    2. HML : 82% correlation
    3. RMW : 77% correlation
    4. CMA : 41% correlation
    
The high correlation of the SMB factor means that all my sorting and aggregating functions
are working as intended. The problem is the definition of the Book-to-Market (HML),
Profitability (RMW) and Investment (CMA) variables in FirmCharacteristics.csv. Exceptional 
care is required to construct these variables. A potential problem is the use of PERMCOs instead
of PERMNOs?

"""






import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BYearEnd
from pandas.tseries.offsets import BMonthEnd
from functools import reduce
import matplotlib.pyplot as plt
# Python time counter 
from time import perf_counter


# Main directory
wdir = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\FamaFrench2015FF5'
os.chdir(wdir)
# Portfolio directory 
FFDIR = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\FamaFrench2015FF5\FFDIR'


# Import the PortSort Class. For more details: 
# https://github.com/ioannisrpt/PortSort.git 
from PortSort import PortSort




# -------------------------------------------------------------------------------------
#                FUNCTIONS - START
# ------------------------------------------------------------------------------------


def WeightedMean(x, df, weights):
    """
    Define the weighted mean function
    """
    return np.average(x, weights = df.loc[x.index, weights])


    
    
    
def JuneScheme(x, num_format = False):
    """
    Use the June-June scheme as in Fama-French.
    
    x must be a datetime object
    """
    # Get month and year
    month = x.month
    year = x.year
    if month <= 6:
        # New date in string format 
        june_dt = '%s-06-01' % year
        y = pd.to_datetime(june_dt, format = '%Y-%m-%d')
        if num_format:
            return BMonthEnd().rollforward(y).strftime('%Y%m%d')
        else:
            return BMonthEnd().rollforward(y)
    else:
        nyear = year + 1
        # New date in string format 
        june_dt = '%s-06-01' % nyear
        y = pd.to_datetime(june_dt, format = '%Y-%m-%d')
        if num_format:
            return BMonthEnd().rollforward(y).strftime('%Y%m%d')
        else:
            return BMonthEnd().rollforward(y)
        
        
# Function that constructs the portfolios per Fama-French methodology 
# along with their market cap weighted returns.
def FFPortfolios(ret_data, firmchars, entity_id, time_id, ret_time_id, characteristics, lagged_periods, \
                 N_portfolios, quantile_filters, ffdir, conditional_sort = True, weight_col = 'CAP_W'):
    """
    
    Parameters
    ----------
    ret_data : Dataframe
        Dataframe where returns for entities are stored in a panel format.
    firmchars : Dataframe
        The characteristics of the entities in ret_data on date_jun.
    entity_id : str
        Entity identifier as found in both ret_data and firmchars.
    time_id : str
        Time identifier as found in both ret_data and firmchars.
    ret_time_id : str
        Time identifier as found in ret_data. ret_time_id dicates the frequency for which market cap
        returns of the portfolios are calculated.
    characteristics : list
        A list of up to three characteristics for which entities will be sorted.
    lagged_periods : list
        A list of the number of lagged periods for the characteristics to be sorted.
        The length of characteristics and lagged_periods must match.
    N_portfolios : list 
        N_portfolios is a list of n_portfolios.
        If n_portfolios then stocks will be sorted in N_portfolios equal portfolios.
        If n_portfolios is an array then this array represents the quantiles. 
    quantile_filters : list
        It is a list of lists. Each element corresponds to filtering entities for the 
        ranking of portfolios into each firm characteristic. The lenght of the list must 
        be the same as that of firm_characteristics.
    ffdir : directory
        Saving directory.
    conditional_sort : boolean, Default=True
        If True, all sorts are conditional. If False, all sorts are unonconditional and 
        independent of each other. 
    weight_col : str, Default='CAP_W'
        The column used for weighting the returns in a portfolio. Default is 'CAP_W' which
        corresponds to the market capitalization of the previous period as defined by 'time_id'.

    Returns
    -------
    port_dict : dictionary
        Directory with items:
            'ports' = portfolio returns
            'num_stocks' = number of stocks in each portfolio
            

    """
    
    # Drop observations if CAP_W is null
    firmchars = firmchars.dropna(subset = [weight_col])
    
    # Define the class using the first sorting characteristic
    port_char = PortSort(firmchars, firm_characteristic = characteristics[0], \
                         lagged_periods = lagged_periods[0], n_portfolios = N_portfolios[0], \
                         entity_id = entity_id, time_id = time_id, quantile_filter = quantile_filters[0], \
                         save_dir = ffdir)
    
    # -----------------------------------
    #  SORT -- SINGLE or DOUBLE or TRIPLE
    # -----------------------------------
    
    # One characteristic --> Single Sort
    # ----------------------------------
    if len(characteristics) == 1:
        
        # Univariate sort
        port_char.SingleSort()  
        
        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.portfolio
        ports = port_char.single_sorted[[time_id, entity_id, weight_col, port_name]].copy()    
        
        
        # Define save names
        save_str =  '%d_portfolios_sortedBy_%s.csv' % (port_char.num_portfolios, characteristics[0])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
        
    # Two characteristic --> Double Conditional Sort
    # -----------------------------------------------
    if len(characteristics) == 2:
                              
        # Bivariate sort
        port_char.DoubleSort(characteristics[1], lagged_periods[1], N_portfolios[1], quantile_filter_2 = quantile_filters[1], \
                             conditional = conditional_sort, save_DoubleSort = False)   
        
        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.double_sorted.columns[-1]
        ports = port_char.double_sorted[[time_id, entity_id, weight_col, port_name]].copy()
        
        # Define save names
        save_str =  '%dx%d_portfolios_sortedBy_%s_and_%s.csv' % ( port_char.num_portfolios, \
                                                                    port_char.num_portfolios_2, \
                                                                    characteristics[0], characteristics[1])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
        
    
    # Three characteristics --> Triple Conditional Sort
    # -------------------------------------------------
    if len(characteristics) == 3:
        
        # Triple sort
        port_char.TripleSort(characteristics[1], characteristics[2], lagged_periods[1], lagged_periods[2], \
                             n_portfolios_2  = N_portfolios[1], n_portfolios_3 = N_portfolios[2], \
                             quantile_filter_2 = quantile_filters[1], quantile_filter_3 = quantile_filters[2], \
                             conditional = conditional_sort, save_TripleSort = False)

        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.triple_sorted.columns[-1]
        ports = port_char.triple_sorted[[time_id, entity_id, weight_col, port_name]].copy()       

        
        # Define save names
        save_str =  '%dx%dx%d_portfolios_sortedBy_%s_and_%s_and_%s.csv' % ( port_char.num_portfolios, \
                                                                    port_char.num_portfolios_2, \
                                                                    port_char.num_portfolios_3,
                                                                    characteristics[0], characteristics[1],\
                                                                    characteristics[2])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
            

    
    
    # Number of stocks in a portfolio
    # -------------------------------
    num_stocks = ports.groupby(by = [port_name, port_char.time_id] )[port_name].count().unstack(level=0)
    
    
    # --------------------------------------------------
    #  ASSIGN PORTFOLIOS TO RETURN DATA (MONTHLY OR DAILY)
    # --------------------------------------------------
    
    # The inner merging is taking care of stocks that should be excluded from the formation of the portfolios
    ret_ports = pd.merge(ret_data, ports, how = 'inner', on = [time_id, entity_id], suffixes = ('', '_2') )
    
    char_ports = ret_ports.groupby(by = [port_name, ret_time_id] ).agg( { 'RET' : lambda x: WeightedMean(x, df = ret_ports, weights = weight_col) } ).unstack(level=0)
    # Rename the columns by keeping only the second element of their names
    char_ports.columns = [x[1] for x in char_ports.columns]
    
    #-------------
    # SAVE RESULTS
    # ------------
            
    char_ports.to_csv(os.path.join(ffdir, save_ret ))
    num_stocks.to_csv(os.path.join(ffdir, save_num ))
    
    # Put everyting in a dictionary
    port_dict = {}
    port_dict['ports'] = char_ports
    port_dict['num_stocks'] = num_stocks
    

    return port_dict


# -------------------------------------------------------------------------------------
#                FUNCTIONS - END
# ------------------------------------------------------------------------------------






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      IMPORT-FORMAT DATA                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------
# CRSP data
# ---------

# Import CRSP monthly data used for extracting the returns of the Fama-French factors
crsp = pd.read_csv(os.path.join(wdir, 'CRSPmonthlydata1963.csv'))
# Keep only certain columns
crspm = crsp[['PERMCO', 'date', 'RET', 'date_jun']].copy()
del crsp

# Show the format of crspm
print(crspm.head(15))

# ----------
# MAP DATES
# ---------

# Isolate the monthly dates of CRSP (last trading day of the month)
mdates = pd.DataFrame(data = crspm['date'].drop_duplicates().sort_values().reset_index(drop = True), columns = ['date'])
# Define a new  date column with only the year-month
mdates['dateYM'] = pd.to_datetime(mdates['date'], format = '%Y%m%d').apply(lambda x: x.strftime('%Y%m')).astype(np.int64)

# Show the format of mdates
print(mdates.head(15))

# --------------------
# FIRM CHARACTERISTICS
# --------------------

# Import FirmCharacteristics table used for sorting stocks in portfolios
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristics.csv'))

# Show the format of firmchars
print(firmchars.head(15))

# -------------------
# FAMA-FRENCH FACTORS
# -------------------

# Import original Fama-French factors from their site
ff5 = pd.read_csv(os.path.join(wdir, 'F-F_Research_Data_5_Factors_2x3.csv'))
# Divide by 100 because returns are in percentage form
ffcols = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
ff5[ffcols] = ff5[ffcols]/100
# Rename 
ff5 = ff5.rename(columns = {'date' : 'dateYM'})
# Use mdates dataframe columns to match year-month to year-month-date
ff5 = pd.merge(ff5, mdates, how = 'left', on = 'dateYM')

# Show the format of firmchars
print(ff5.head(15))

# Save ff5 for future use
ff5.to_csv(os.path.join(wdir, 'FF5_monthly.csv'), index = False)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        SMB FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2 Size portfolios
size= FFPortfolios(crspm, firmchars, entity_id = 'PERMCO', time_id = 'date_jun', \
                       ret_time_id = 'date', characteristics = ['CAP'], lagged_periods = [1], \
                       N_portfolios = [2 ], quantile_filters = [['EXCHCD', 1]], \
                       ffdir = FFDIR, conditional_sort = False)
    
# Renaming the portfolios as per Fama-French definition
# Size : 1 = Small, 2 = Big
size_def = {1 : 'S', 2 : 'B'}


    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
size_p = size['ports'][size['ports'].index > size['num_stocks'].index[0]].copy().rename(columns = size_def)

# Define the SMB factor
size_p['mySMB']  = size_p['S'] - size_p['B']


# Comparison of mySMB and SMB
smb_comp = pd.merge(size_p['mySMB'], ff5.set_index('date')['SMB'], how = 'inner', on = 'date')
print('--- Comparison of mySMB and Fama-French SMB factor ----  \n')
print(smb_comp.corr())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        HML FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2x3 Size and Book-to-Market portfolios 
sizebtm = FFPortfolios(crspm, firmchars, entity_id = 'PERMCO', time_id = 'date_jun', \
                       ret_time_id = 'date', characteristics = ['CAP', 'BtM'], lagged_periods = [1,1], \
                       N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                       ffdir = FFDIR, conditional_sort = False)

    
# Renaming the portfolios as per Fama-French definition
# Size : 1 = Small, 2 = Big
# BtM : 1 = Low, 2 = Neutral, 3 = High
sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
               '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizebtm_p = sizebtm['ports'][sizebtm['ports'].index > sizebtm['num_stocks'].index[0]].copy().rename(columns = sizebtm_def)

# Define the HML factor
sizebtm_p['myHML'] = (1/2)*(sizebtm_p['SH'] + sizebtm_p['BH']) - \
                     (1/2)*(sizebtm_p['SL'] + sizebtm_p['BL'])  
                     
# Comparison of myHML and HML
hml_comp = pd.merge(sizebtm_p['myHML'], ff5.set_index('date')['HML'], how = 'inner', on = 'date')
print('--- Comparison of myHML and Fama-French HML factor ----  \n')
print(hml_comp.corr())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        RMW FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the 2x3 Size and Profitability portfolios 
sizermw = FFPortfolios(crspm, firmchars, entity_id = 'PERMCO', time_id = 'date_jun', \
                       ret_time_id = 'date', characteristics = ['CAP', 'OP'], lagged_periods = [1,1], \
                       N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                       ffdir = FFDIR, conditional_sort = False)

    
# Renaming the portfolios as per Fama-French definition
# Size : 1 = Small, 2 = Big
# OP : 1 = Weak, 2 = Neutral, 3 = Robust
sizermw_def = {'1_1' : 'SW', '1_2' : 'SN', '1_3' : 'SR', \
               '2_1' : 'BW', '2_2' : 'BN', '2_3' : 'BR'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizermw_p = sizermw['ports'][sizermw['ports'].index > sizermw['num_stocks'].index[0]].copy().rename(columns = sizermw_def)

# Define the RMW factor
sizermw_p['myRMW'] = (1/2)*(sizermw_p['SR'] + sizermw_p['BR']) - \
                     (1/2)*(sizermw_p['SW'] + sizermw_p['BW'])  
                     
# Comparison of myRMW and RMW
rmw_comp = pd.merge(sizermw_p['myRMW'], ff5.set_index('date')['RMW'], how = 'inner', on = 'date')
print('--- Comparison of myRMW and Fama-French RMW factor ----  \n')
print(rmw_comp.corr())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                        CMA FACTOR                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Create the 2x3 Size and Investment portfolios 
sizecma = FFPortfolios(crspm, firmchars, entity_id = 'PERMCO', time_id = 'date_jun', \
                       ret_time_id = 'date', characteristics = ['CAP', 'INV'], lagged_periods = [1,1], \
                       N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                       ffdir = FFDIR, conditional_sort = False)

    
# Renaming the portfolios as per Fama-French definition
# Size : 1 = Small, 2 = Big
# INV : 1 = Conservative, 2 = Neutral, 3 = Aggressive
sizecma_def = {'1_1' : 'SC', '1_2' : 'SN', '1_3' : 'SA', \
               '2_1' : 'BC', '2_2' : 'BN', '2_3' : 'BA'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizecma_p = sizecma['ports'][sizecma['ports'].index > sizecma['num_stocks'].index[0]].copy().rename(columns = sizecma_def)

# Define the CMA factor
sizecma_p['myCMA'] = (1/2)*(sizecma_p['SC'] + sizecma_p['BC']) - \
                     (1/2)*(sizecma_p['SA'] + sizecma_p['BA'])  
                     
# Comparison of myCMA and CMA
cma_comp = pd.merge(sizecma_p['myCMA'], ff5.set_index('date')['CMA'], how = 'inner', on = 'date')
print('--- Comparison of myCMA and Fama-French CMA factor ----  \n')
print(cma_comp.corr())




# ----------------
# SAVE ALL FACTORS
# ----------------

factors_ls =[smb_comp, hml_comp, rmw_comp, cma_comp]

allfactors = reduce(lambda a,b: a.join(b), factors_ls)

# Save it
allfactors.reset_index().to_csv(os.path.join(wdir, 'ALLFACTORS_monthly.csv'), index = False)
