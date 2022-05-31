import pandas as pd
from tweet_coding.tweet_coding import filter_coding, hazard_risk_coding, irma_image_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'irma_ic_tocode.csv'
fcoded = 'irma_ic_coded.csv'
d = 'Irma\\Data\\Image Coding'
datecols = ['created_at']

irma_image_coding(cfile=fcoded,
                   dfile=f,
                   direc=d,
                   ctype='Image coding',
                   datecols=datecols,
                   url_col='tweet-url')


# hazard_risk_coding(cfile=fcoded,
#                    dfile=f,
#                    direc=d,
#                    ctype='Hazard and risk information',
#                    datecols=datecols,
#                    url_col='tweet-url')

#filter_coding(cfile=fcoded,
#                dfile=f,
#                direc=d,
#                ctype="Initial filtering",
#                codecols=['deleted_qt', 'relevant', 'spanish'],
#                datecols=datecols)
