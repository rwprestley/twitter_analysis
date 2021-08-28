import pandas as pd
from tweet_coding.tweet_coding import filter_coding, hazard_risk_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'irma_rel_samp2_100.csv'
fcoded = 'irma_rel_samp2_100_coded_rp.csv'
d = 'Irma\\Data\\Content Coding - Phase 1'
datecols = ['created_at']

hazard_risk_coding(cfile=fcoded,
                   dfile=f,
                   direc=d,
                   ctype='Hazard and risk information',
                   datecols=datecols,
                   url_col='tweet-url')

#filter_coding(cfile=fcoded,
#                dfile=f,
#                direc=d,
#                ctype="Initial filtering",
#                codecols=['deleted_qt', 'relevant', 'spanish'],
#                datecols=datecols)
