import pandas as pd
from tweet_coding.tweet_coding import rel_fore_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'irma_tweets_samp_100.csv'
fcoded = 'irma_tweets_samp_100_coded.csv'
d = 'Irma\\Data'
datecol = 'created_at'

rel_fore_coding(cfile=fcoded,
                dfile=f,
                direc=d,
                ctype="Initial filtering",
                codecols=['deleted_qt', 'relevant', 'spanish', 'forecast', 'local_relevant'],
                datecol=datecol)
