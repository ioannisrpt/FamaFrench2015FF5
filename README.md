# Replication of Fama & French (2015) - A five factor asset pricing model
I replicate the five Fama-French factors on the monthly frequency. 
The results of my replication in terms of correlation are as follows:
1. SMB: 97.19% 
2. HML: 95.15%
3. RMW: 91.81%
4. CMA: 97.30%

The "Cumulative returns" folder contains a visual comparison of the replicated factors
with the original by plotting their cumulative returns from July 1963 to December 2021. 

# FormatCRSPdata.py
I use the CRSPSift application for Windows to extract the following data from the 
CRSP monthly tape:
1. RET: total return 
2. PRC: end-of-period price
3. SHROUT: number of shares outstanding 
4. EXCHCD: exchange code
5. SHRCD: security share code 
6. PERMNO: security identifier
7. PERMCO: company identifier

I extract everything for the period Jan 1926 to Dec 2021 and then I subset it.

# CreateFirmCharacteristicsFF5Dataset.py
I extract all data necessary to construct Book-to-Market, Operating Profitability 
and Investability variable as described in the paper from Compustat. The only filter 
I use is that items must be reported in USD. Then I proceed to define the aforementioned 
variables and merge them with market equity data from CRSP. 

# FamaFrench2015FF5.py
I subset the data to include only common ordinary shares (SHRCD = 10, 11) that trade in
NYSE, AMEX and NASDAQ (EXCHCD = 1, 2, 3) after June 1963. Then I proceed to construct 
the factors using the methods of PortSort. Check https://github.com/ioannisrpt/portsort 
for more details. 
