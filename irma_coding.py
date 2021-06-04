import pandas as pd
from tweet_coding.tweet_coding import rel_fore_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'irma_to_code_robert.csv'
fcoded = 'irma_coded_robert.csv'
d = 'Irma\\Data'
datecols = ['user.username', 'created_at']

rel_fore_coding(cfile=fcoded,
                dfile=f,
                direc=d,
                ctype="Initial filtering",
                codecols=['deleted_qt', 'relevant', 'spanish'],
                datecols=datecols)
