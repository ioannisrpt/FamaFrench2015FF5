# FamaFrench2015FF5
Replication of the five Fama-French factors for the sample period July 1965 to July 2020 using
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
