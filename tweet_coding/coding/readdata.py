import os
import pandas as pd


def readdata(cfile, dfile, direc, ctype, codecols, datecol, **kwargs):
    """
    Read data from a Twitter database

    Parameters:
        cfile: A text string denoting what the coded data should be saved as
        dfile: A text string denoting the existing data that should be coded
        direc: A text string denoting the directory that the existing data and newly coded data should be saved in
        ctype: A text string denoting the type of coding being done (e.g. 'Risk image', 'Relevance and forecast'). First
                  letter should be capitalized.
        codecols: A list of strings, where each string represents a column where coded data is stored
        datecol: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
    """
    # If coding file does not exist yet, coding has not been initiated.
    if (cfile in os.listdir(direc)) is not True:
        print(ctype + ' coding has not been initiated')

        # Create a dataframe with tweets to be coded.
        to_code = pd.read_csv(direc + '\\' + dfile)

        # Select tweets to be coded based on criteria, if provided
        select_col = kwargs.get('select_col')
        select_crit = kwargs.get('select_crit')

        if (select_col is not None) & (select_crit is not None):
            to_code = to_code.loc[to_code[select_col] == select_crit]

        # Initialize columns where codes should be stored.
        for col in codecols:
            to_code[col] = ''

        to_code.sort_values(datecol, inplace=True)

        # Create an empty coded dataframe.
        coded = pd.DataFrame()

    # If coding file does exist, coding has already started and should be continued where it left off.
    else:
        print(ctype + ' coding already started')

        # Read and clean the coded data file.
        tweetdata = pd.read_csv(direc + '\\' + cfile)
        for col in tweetdata.columns:
            if col[:7] == 'Unnamed':
                tweetdata.drop(col, axis=1, inplace=True)

        for col in codecols:
            tweetdata[col] = tweetdata[col].fillna('')

        # Split the data in to tweets that have already been coded and tweets that have yet to be coded.
        coded = tweetdata.loc[tweetdata[codecols[0]] != '']
        to_code = tweetdata.loc[tweetdata[codecols[0]] == '']

        if len(to_code) == 0:
            print(ctype + ' coding completed')
        else:
            print(ctype + ' coding progress: ' + str(len(coded)) + '/' + str(len(tweetdata)) +
                  ' (' + '{:.2%}'.format(len(coded) / len(tweetdata)) + ')')

    return coded, to_code
