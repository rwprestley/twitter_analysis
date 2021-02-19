import pandas as pd


def save_coding(coded, to_code, cfile, direc, datecol):
    """
        Merge coded and uncoded data together and save

        Parameters:
            coded: A Pandas dataframe of tweets that have already been coded
            to_code: A Pandas dataframe of tweets to be coded
            cfile: A text string denoting what the coded data should be saved as
            direc: A text string denoting the directory that the existing data and newly coded data should be saved in
            datecol: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
        """
    tweetdata = pd.concat([coded, to_code])
    tweetdata.sort_values(by=datecol, inplace=True)
    tweetdata.to_csv(direc + '\\' + cfile)

    return tweetdata
