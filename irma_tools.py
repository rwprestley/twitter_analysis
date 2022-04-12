import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns


def google_coded_format(fileloc, to_rename, renamed, header):
    """
    Format tweet spreadsheets created in Google Sheets to code tweet data

    Parameters:
        fileloc: File name/location in directory (string)
        to_rename: Names of columns to rename, corresponding to final coded columns (list of strings)
        renamed: New names for columns to be renamed (list of strings)
        header: The row that the column headers are stored in (integer)

    Returns:
        A Pandas dataframe of tweet metadata (id and tweet-url) and coded data

    Outputs:
        None
    """

    # Read in files and only keep tweet-id, tweet-url, and coded columns
    df = pd.read_csv(fileloc, header=header)
    keep_cols = ['id', 'tweet-url'] + to_rename
    df = df[keep_cols]

    # Rename columns
    df.rename(columns=dict(zip(to_rename, renamed)), inplace=True)

    # Reformat ids
    df['id'] = df['tweet-url'].str[-18:]

    # Drop duplicates
    df = df.drop_duplicates('id')

    return df


def join_vals(df, col_names, val, prefix_len):
    """
    Concatenates column names where column values in a row are equal to a provided value

    Parameters:
        df: A Pandas dataframe representing one row of data (used in a dataframe "apply" function to calculate
            over an entire dataframe)
        col_names: Columns to concatenate together (if column values meet the provided value)
        val: Value that column values should meet in order for their names to be concatenated together
        prefix_len: How many values to strip from the beginning of the column names to remove prefixes

    Returns:
        Concatenated column values (string)

    Outputs:
        None
    """

    # Select columns to concatenate together by searching for where the values are equal to the provided value
    col_keep = []
    for col_num, col_name in enumerate(col_names):
        if df[col_names].iloc[col_num] == val:
            col_keep.append(col_name)

    # Clean column names by removing prefix
    col_keep_clean = [c[prefix_len:] for c in col_keep]

    # Concatenate column names together
    concatenated = ' & '.join(col_keep_clean)

    return concatenated


def haz_risk_format(df):
    """
    Create mutually exclusive hazard & risk information columns from coded data

    Parameters:
        df: A Pandas dataframe of Twitter data

    Returns:
        The same dataframe as entered, with updates to hazard & risk columns as described above

    Outputs:
        None
    """
    # Convert coded hazard and risk information columns from "yes" and "no to 1s and 0s
    haz_names = ['hazard_tc', 'hazard_surge', 'hazard_rain_flood', 'hazard_convective', 'hazard_mult', 'hazard_other']
    risk_names = ['risk_non_ww_fore', 'risk_ww_fore', 'risk_obs', 'risk_past', 'risk_mult', 'risk_other']
    for col in (haz_names + risk_names):
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Count the number of hazard and risk information codes applied (not counting multiple or other columns)
    df['hazard_count'] = df[haz_names[0:4]].sum(axis=1)
    df['risk_count'] = df[risk_names[0:4]].sum(axis=1)

    # Manually define multiple columns
    df[['hazard_mult', 'risk_mult']] = 0
    df.loc[df['hazard_count'] > 1, 'hazard_mult'] = 1
    df.loc[df['risk_count'] > 1, 'risk_mult'] = 1

    # Manually define other columns
    df[['hazard_other', 'risk_other']] = 0
    df.loc[df['hazard_count'] == 0, 'hazard_other'] = 1
    df.loc[df['risk_count'] == 0, 'risk_other'] = 1

    # Create combined hazard and risk information code columns
    df['hazard_join'] = df.apply(lambda row: join_vals(row, haz_names[:4] + ['hazard_other'], 1, 7), axis=1)
    df['risk_join'] = df.apply(lambda row: join_vals(row, risk_names[:4] + ['risk_other'], 1, 5), axis=1)

    # Create lump multiple hazard and risk information code columns
    df['hazard'] = df[haz_names + ['hazard_other']].idxmax(axis=1).str[7:]
    df.loc[df['hazard_count'] > 1, 'hazard'] = 'multiple'

    df['risk'] = df[risk_names + ['risk_other']].idxmax(axis=1).str[5:]
    df.loc[df['risk_count'] > 1, 'risk'] = 'multiple'

    # Create hazard/risk crosstab codes
    df['hazard_risk'] = df['hazard'] + '_' + df['risk']

    return df


def merge_coding(filter_dfs, haz_risk_dfs, raw_df):
    """
    Merge uncoded data from Twitter API with data coded by research team

    Parameters:
        filter_dfs: A list of Irma filter-coded dataframes to concatenate
        haz_risk_dfs: A list of Irma hazard and risk information-coded dataframes to concatenate
        raw_df: A dataframe with the original data pulled from the API, to merge coded data with

    Returns:
        A Pandas dataframe of raw Twitter data merged with coded data

    Outputs:
        None
    """

    # Concatenate filter coded data together
    irma_filter = pd.concat(filter_dfs, join='outer')

    # Concatenate hazard and risk coded data together
    irma_haz_risk = pd.concat(haz_risk_dfs, join='outer')

    # Format hazard & risk codes/columns
    irma_haz_risk = haz_risk_format(irma_haz_risk)

    # Drop unnecessary columns
    for col in raw_df.columns:
        if col[:7] == 'Unnamed':
            raw_df.drop(col, axis=1, inplace=True)

    for col in irma_filter.columns:
        if col[:7] == 'Unnamed':
            irma_filter.drop(col, axis=1, inplace=True)

    # Merge hazard and risk coded data with filter coded data
    merged1 = pd.merge(irma_filter, irma_haz_risk, how='left', on='tweet-url', suffixes=('', '_y'))
    merged1.drop(merged1.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    merged1['id'] = merged1['tweet-url'].str[-18:]

    # Merge coded data with uncoded raw data
    merged2 = pd.merge(raw_df, merged1, how='left', on='tweet-url', suffixes=('', '_y'))
    merged2.drop(merged2.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
    merged2['id'] = merged2['tweet-url'].str[-18:]

    # Remove unaccessable tweets from Steve Jerve (@sjervewfla)
    merged2 = merged2[merged2['user.username'] != 'sjervewfla']

    return merged2


def text_format(df):
    """
    Format the tweet text in a Pandas dataframe

    Parameters:
        df: A Pandas dataframe of Twitter data (must include the following columns:
            'text', 'source', 'tweet-url', and 'user.username'

    Returns:
        The same dataframe input, but with updated and newly created text columns

    Outputs:
        None
    """

    # Split URLs from the rest of the text
    df.loc[df['text'].str[-23:-18] == 'https', 'text_link'] = df['text'].str[-23:]
    df.loc[df['text'].str[-23:-18] == 'https', 'text'] = df['text'].str[:-23]

    # Count the number of duplicate text tweets within each user, for the full text and just the first 15 characters
    df['text_source_count'] = df.groupby(['text', 'source'])['tweet-url'].transform('count')

    df['text_15'] = df['text'].str[:15]
    df['user_source_text_15_count'] = df.groupby(['user.username', 'source', 'text_15'])['tweet-url'].transform('count')

    # Format text for natural language processing and text searching
    df['text_lower'] = df['text'].str.lower()
    df['text_clean'] = df['text_lower'].str.replace('[^a-zA-Z ]', ' ', regex=True)

    return df


def bot_coding(df):
    """
    Performs coding of repeated text phrases/bot-like content

    Parameters:
        df: A Pandas dataframe of Twitter data (text, source, and user columns must be named 'text',
            'source', and 'user.username', respectively)

    Returns:
        The same dataframe as entered, with a 'bot_tweet' column which includes bot coding results

    Outputs:
        None
    """

    # Coding bot sources
    bot_sources = ['Fox 13 Weather Alerts', 'NWSBot', 'NWSWPC_AutoTweet', 'Severe Wx Impact Graphics - MFL',
                   'SPC Tweepy Update', 'Svr Wx Impact Graphics - TBW', 'TTYtter']
    for bot_source in bot_sources:
        df.loc[df['source'] == bot_source, 'bot_tweet'] = 1

    # Coding bot text strings
    df.loc[df['text'].str[:8] == 'Close-up', 'bot_tweet'] = 1
    df.loc[df['text'].str[:16] == '10 Weather Alert', 'bot_tweet'] = 1
    df.loc[df['text'].str[:10] == 'Vientos mi', 'bot_tweet'] = 1
    df.loc[df['text'].str[:33] == 'Here is the latest forecast track', 'bot_tweet'] = 1
    df.loc[df['text'].str[:20] == 'Here are the lastest', 'bot_tweet'] = 1
    df.loc[df['text'].str[-15:] == 'and on our APP ', 'bot_tweet'] = 1

    # Coding variable bot text strings
    hazards = ['TORNADO', 'FLASH FLOOD', 'FLOOD']
    alert_type = ['WATCH', 'WARNING']
    for hazard in hazards:
        for alert in alert_type:
            loc_string = hazard + ' ' + alert + ' for parts of'
            str_len = len(loc_string)
            df.loc[df['text'].str[:str_len] == loc_string, 'bot_tweet'] = 1

    # Coding bot user/source combinations
    bot_users = ['daveofox13', 'fox13tyler', 'jimweberfox', 'paulfox13', 'weatherlindsay']
    for bot_user in bot_users:
        df.loc[(df['user.username'] == bot_user) & (df['source'] == 'SocialNewsDesk'), 'bot_tweet'] = 1

    # Coding non-bot tweets
    df.loc[df['bot_tweet'] != 1, 'bot_tweet'] = 0

    return df


def engage_calc(df, engage_types):
    """
    Calculates engagement metrics for provided engagement types (e.g. retweet, reply, like, quote)

    Parameters:
        df: A Pandas dataframe of Twitter data
        engage_types: A list of engagement types in the Twitter data

    Returns:
        The same dataframe as entered, with included engagement metric columns

    Outputs:
        None

    Notes:
        The engage_types must match the roots of the engagement columns ending in 'count' received from the Twitter
        V2 API (e.g. retweet, reply, like, quote).
        This function also log-normalizes follower counts; thus, the dataframe should include a column named
        'user.followers_count'.
    """
    # Log-normalizing engagement and user characteristics
    for engage in engage_types:
        df[engage + '_plus1'] = df[engage + '_count'] + 1
        df[engage + '_log_norm'] = np.log10(df[engage + '_plus1'])
    df['follower_log_norm'] = np.log10(df['user.followers_count'] + 1)

    # Calculating total engagement (RT + QT + replies + likes) and log normalizing
    df['engage_count'] = df['retweet_count'] + df['quote_count'] + df['reply_count'] + df['like_count']
    df['engage_plus1'] = df['engage_count'] + 1
    df['engage_log_norm'] = np.log10(df['engage_count'] + 1)

    return df


def user_format(df):
    """
    Merges coded authoritative source/user data with API data

    Parameters:
        df: A Pandas dataframe of Twitter data

    Returns:
        The same dataframe as entered, with coded user/source coded characteristics (scope, local scope, agency,
            affiliation)

    Outputs:
        None
    """
    # Merge authoritative source codes with full Twitter data
    irma_sources = pd.read_csv('Irma\\Data\\irma_sources.csv')
    source_cols = ['Originator (API)', 'Scope', 'Local Scope', 'Agency', 'Affiliation']
    new_source_cols = ['user.username', 'user.scope', 'user.local_scope', 'user.agency', 'user.affiliation']
    irma_sources = irma_sources[source_cols].rename(dict(zip(source_cols, new_source_cols)), axis=1)
    df = df.merge(right=irma_sources, how='left', on='user.username')

    # Merge local scope with overall scope
    df.loc[df['user.local_scope'] == 'Miami/Fort Lauderdale', 'user.scope'] = 'Miami'
    df.loc[df['user.local_scope'] == 'Tampa/St Petersburg', 'user.scope'] = 'Tampa'
    df.loc[df['user.scope'] == 'National/International', 'user.scope'] = 'National'

    # Merge scope and affiliation codes
    df['user.scope_aff'] = df['user.scope'] + ' ' + df['user.affiliation']

    return df


def micro_format(df):
    """
    Creates counts/inclusion columns for key microstructural features (hashtags, mentions, and URLs)

    Parameters:
        df: A Pandas dataframe of Twitter data

    Returns:
        The same dataframe as entered, with counts/inclusion columns for key microstructural features

    Outputs:
        None
    """
    # Hashtag counts/inclusion
    hash_cols = [col for col in df.columns if (col[:7] == 'hashtag') & (col[-3:] == 'tag')]
    df['hashtags.count'] = df[hash_cols].count(axis=1)
    df.loc[df['hashtags.count'] > 0, 'includes.hashtag'] = 1
    df.loc[df['hashtags.count'] == 0, 'includes.hashtag'] = 0

    # Mention counts/inclusion
    mention_cols = [col for col in df.columns if (col[:8] == 'mentions') & (col[-8:] == 'username')]
    df['mentions.count'] = df[mention_cols].count(axis=1)
    df.loc[df['mentions.count'] > 0, 'includes.mention'] = 1
    df.loc[df['mentions.count'] == 0, 'includes.mention'] = 0

    # URL counts/inclusion
    url_cols = [col for col in df.columns if (col[:4] == 'urls') & (col[-4:] == '.url')]
    df['urls.count'] = df[url_cols].count(axis=1) - df['media.count']
    df.loc[df['urls.count'] > 0, 'includes.url'] = 1
    df.loc[df['urls.count'] == 0, 'includes.url'] = 0

    return df


def ling_format(df):
    """
    Creates counts/inclusion columns for key linguistic text features (!, ?, ALL CAPS)

    Parameters:
        df: A Pandas dataframe of Twitter data

    Returns:
        The same dataframe as entered, with counts/inclusion columns for key linguistic text features

    Outputs:
        None
    """
    # Counts/inclusion of ! and ? within tweet text
    df['!'] = np.where(df['text'].str.contains('!'), 1, 0)
    df['?'] = np.where(df['text'].str.contains('?', regex=False), 1, 0)

    # Counts/inclusions of ALL CAPS within tweet text
    df['utext_clean'] = df['text'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
    df['CAP_count'] = df.apply(lambda row: sum(map(str.isupper, row['utext_clean'].split())), axis=1)
    df.loc[df['CAP_count'] > 0, 'includes.CAP'] = 1
    df.loc[df['CAP_count'] == 0, 'includes.CAP'] = 0

    return df


def dummy_code(df, col):
    """
    Create dummy codes from a Pandas dataframe

    Parameters:
        df: A Pandas dataframe
        col: Column with categorical data to create dummy variables from

    Returns:
        The same dataframe as entered, but with new dummy variable columns

    Outputs:
        None
    """
    # Create dummy codes from a categorical variable
    for cat in df[col].drop_duplicates().tolist():
        dummy_col = col + '_' + str(cat)
        df.loc[df[col] == cat, dummy_col] = 1
        df.loc[df[col] != cat, dummy_col] = 0

    return df


def time_format(df, timezone):
    """
       Cuts/slices & formats times into groups, and creates dummy variables for time/date categories

       Parameters:
           df: A Pandas dataframe of Twitter data
           timezone: Datetime timezone object corresponding to local time of interest (e.g. Eastern, Central)

       Returns:
           The same dataframe as entered, with time slice & date/time dummy columns included

       Outputs:
           None
    """
    # Convert date/time to timezone-aware datetime format
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.tz_convert('America/New_York')

    # Pull dates and hour of day into seperate columns
    df['date'] = df['created_at'].dt.date
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%m-%d')
    df['hour'] = df['created_at'].dt.hour

    # Create time of day column (adding time to a neutral date - 9/1)
    df['time_ofday'] = '2017-09-01 ' + df['created_at'].dt.time.astype(str)
    df['time_ofday'] = pd.to_datetime(df['time_ofday'])
    df['time_ofday'] = df['time_ofday'].dt.tz_localize(timezone)

    # Create a 6-hour binned date/time column
    bins_hour = [0, 6, 12, 18, 24]
    labels_hour = ['3:00', '9:00', '15:00', '21:00']
    df['created_at_6h'] = pd.cut(df['created_at'].dt.hour, bins=bins_hour, labels=labels_hour, right=False)
    df['created_at_date6h'] = df['created_at'].dt.date.astype(str) + ' ' + df['created_at_6h'].astype(str)
    df['created_at_date6h'] = pd.to_datetime(df['created_at_date6h']).dt.tz_localize(timezone)

    # Bin hours into 3-hour groups
    hbins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    hlabels = ['12-3A', '3-6A', '6-9A', '9A-12P', '12-3P', '3-6P', '6-9P', '9P-12A']
    df['hour_3h'] = pd.cut(df['hour'], bins=hbins, labels=hlabels, right=False)

    # Bin dates into multi-day groups
    dbins = [datetime(2017, 8, 29), datetime(2017, 9, 5), datetime(2017, 9, 9), datetime(2017, 9, 10),
             datetime(2017, 9, 11), datetime(2017, 9, 13)]
    dbins_tz = [timezone.localize(dt) for dt in dbins]
    dlabels = ['Aug30-Sep4', 'Sep5-8', 'Sep9', 'Sep10', 'Sep11-12']
    df['date_range'] = pd.cut(df['created_at'], bins=dbins_tz, labels=dlabels, right=False)

    # Create dummy codes for grouped date and time variables
    dtcols = ['date', 'hour', 'created_at_date6h', 'hour_3h', 'date_range']
    for col in dtcols:
        df = dummy_code(df, col)

    return df


def readdata(how, save=True):
    """
    Reads in Irma Twitter data, either from an already formatted Twitter database (how = 'read') or from scratch,
        formatting data throughout the process (how = 'calc')

    Parameters:
        how: How to read in the data. Can take arguments of 'read', in which data will be read in from a pre-formatted
                 CSV, or 'how', in which data is read in from raw data and formatted
        save: Whether to save the data after reading it in or calculating it (default True)

    Returns:
        A Pandas dataframe of fully-formatted Twitter data for Hurricane Irma

    Outputs:
        None
    """

    # Read in data from raw data and format
    if how == 'calc':
        # Read in API data
        print('Reading in data...')
        irma_api = pd.read_csv('Twitter-V2-API\\API CSV Output\\irma_out.csv', low_memory=False)

        # Read in and format filter coded data
        irma_coded_robert = pd.read_csv('Irma\\Data\\Filter Coding\\irma_coded_robert.csv')
        irma_coded_tbd = pd.read_csv('Irma\\Data\\Filter Coding\\irma_coded_tbd.csv')
        irma_coded_alyssa = pd.read_csv('Irma\\Data\\Filter Coding\\irma_coded_alyssa.csv')
        irma_coded_new = pd.read_csv('Irma\\Data\\Filter Coding\\irma_original_new_0621_coded.csv')

        filter_to_rename = ['Final', 'Final.1', 'Final.2']
        filter_new_names = ['deleted_qt', 'relevant', 'spanish']
        irma_coded_icr = google_coded_format('Irma\\Data\\Filter Coding\\irma_coded_icr.csv', filter_to_rename,
                                             filter_new_names, 1)
        filter_coded = [irma_coded_robert, irma_coded_tbd, irma_coded_alyssa, irma_coded_icr, irma_coded_new]

        # Read in and format hazard and risk coded data
        icr_to_rename = ['Final', 'Final.1', 'Final.2', 'Final.3', 'Final.4', 'Final.5', 'Final.6', 'Final.7',
                         'Final.8', 'Final.9', 'Final.10', 'Final.11']
        bot_to_rename = ['TC', 'Surge', 'Rain/flood', 'Convective', 'Multiple', 'Other', 'Non-W/W Forecast',
                         'W/W Forecast', 'Observational', 'Past', 'Multiple.1', 'Other.1']
        coded_to_rename = ['tc', 'surge', 'rain/flood', 'convective', 'haz_mult', 'haz_other', 'forecast', 'ww', 'obs',
                           'past', 'risk_mult', 'risk_other']
        haz_names = ['hazard_tc', 'hazard_surge', 'hazard_rain_flood', 'hazard_convective', 'hazard_mult',
                     'hazard_other']
        risk_names = ['risk_non_ww_fore', 'risk_ww_fore', 'risk_obs', 'risk_past', 'risk_mult', 'risk_other']
        irma_haz_risk_icr = google_coded_format('Irma\\Data\\Content Coding - Phase 1\\irma_icr_coded.csv',
                                                icr_to_rename, haz_names + risk_names, 2)
        irma_haz_risk_bot = google_coded_format('Irma\\Data\\Content Coding - Phase 1\\irma_bot_coded.csv',
                                                bot_to_rename, haz_names + risk_names, 2)
        irma_haz_risk_coded = google_coded_format('Irma\\Data\\Content Coding - Phase 1\\irma_rel_coded.csv',
                                                  coded_to_rename, haz_names + risk_names, 0)
        haz_risk_coded = [irma_haz_risk_icr, irma_haz_risk_bot, irma_haz_risk_coded]

        # Merge data sources together
        print('Merging data sources...')
        irma_tweets = merge_coding(filter_dfs=filter_coded, haz_risk_dfs=haz_risk_coded, raw_df=irma_api)

        # Format merged data
        print('Formatting merged data...')

        # Format text
        irma_tweets = text_format(irma_tweets)

        # Format media columns
        irma_tweets.loc[irma_tweets['media.type'].isna(), 'media.type'] = 'text-only'
        irma_tweets.loc[
            (irma_tweets['media.type'] != 'photo') & (irma_tweets['media.type'] != 'text-only'), 'rich_media'] = 1
        irma_tweets.loc[
            (irma_tweets['media.type'] == 'photo') | (irma_tweets['media.type'] == 'text-only'), 'rich_media'] = 0

        # Bot coding and segmenting
        irma_tweets = bot_coding(irma_tweets)
        irma_tweets.loc[
            (irma_tweets['bot_tweet'] == 1) & (irma_tweets['includes.media'] == 'media'), 'bot_media'] = 'bot_media'
        irma_tweets.loc[
            (irma_tweets['bot_tweet'] == 1) & (
                        irma_tweets['includes.media'] == 'non-media'), 'bot_media'] = 'bot_non-media'
        irma_tweets.loc[
            (irma_tweets['bot_tweet'] == 0) & (irma_tweets['includes.media'] == 'media'), 'bot_media'] = 'non-bot_media'
        irma_tweets.loc[
            (irma_tweets['bot_tweet'] == 0) & (
                        irma_tweets['includes.media'] == 'non-media'), 'bot_media'] = 'non-bot_non-media'

        # Calculating diffusion & engagement metrics
        engages = ['retweet', 'reply', 'quote', 'like']
        irma_tweets = engage_calc(df=irma_tweets, engage_types=engages)

        # Merging with coded source/user data
        irma_tweets = user_format(irma_tweets)

        # Formatting microstructural features
        irma_tweets = micro_format(irma_tweets)

        # Formatting linguistic features
        irma_tweets = ling_format(irma_tweets)

        # Create filtering columns
        irma_tweets['time'] = 1
        irma_tweets.loc[irma_tweets['referenced_tweets1.type'] == 'original_tweet', 'original'] = 1
        irma_tweets['relevant'] = irma_tweets['relevant'].map(dict(yes=1, no=0))

        # Format/slice time columns
        tz = pytz.timezone('US/Eastern')
        irma_tweets = time_format(irma_tweets, tz)

        # Define original tweet database and calculate diffusion statistics in reference to this set
        irma_original = irma_tweets[irma_tweets['referenced_tweets1.type'] == 'original_tweet']
        for engage in (engages + ['engage']):
            irma_original = diff_calcs(df=irma_original, ref='all', engage_type=engage)

        # Calculate diffusion statistics in reference to relevant tweets
        irma_rel = irma_original[irma_original['relevant'] == 1]
        for engage in (engages + ['engage']):
            irma_rel = diff_calcs(irma_rel, 'rel', engage)

        # For hazard and risk codes, create a non-relevant code for original tweets that weren't relevant to Irma.
        irma_original.loc[irma_original['relevant'] != 1, 'hazard'] = 'non_rel'
        irma_original.loc[irma_original['relevant'] != 1, 'hazard_non_rel'] = 1
        irma_original.loc[irma_original['relevant'] != 1, 'risk'] = 'non_rel'
        irma_original.loc[irma_original['relevant'] != 1, 'risk_non_rel'] = 1

        # Re-merge relevant tweets with original tweets
        irma_original = pd.merge(irma_original, irma_rel, how='left', on='tweet-url', suffixes=('', '_y'))
        irma_original.drop(irma_original.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
        irma_original['id'] = irma_original['tweet-url'].str[-18:]

        # Re-merge original tweets with all tweets
        irma_tweets = pd.merge(irma_tweets, irma_original, how='left', on='tweet-url', suffixes=('', '_y'))
        irma_tweets.drop(irma_tweets.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)
        irma_tweets['id'] = irma_tweets['tweet-url'].str[-18:]

        irma_tweets = irma_tweets.set_index('id')
        irma_tweets.sort_values('created_at', ascending=True, inplace=True)

    # Read in data from pre-formatted database
    elif how == 'read':
        print('Reading in data...')
        irma_tweets = pd.read_csv('irma_tweets.csv')
        irma_tweets['created_at'] = pd.to_datetime(irma_tweets['created_at'])
        irma_tweets['created_at_date6h'] = pd.to_datetime(irma_tweets['created_at_date6h'])
        irma_tweets['id'] = irma_tweets['tweet-url'].str[-18:]

    else:
        irma_tweets = None

    # Save data, if desired
    if save is True:
        irma_tweets.to_csv('irma_tweets_test.csv')

    return irma_tweets


def count_diff(df, ref, strat, **kwargs):
    """
    Create a plot of tweet counts and diffusion over user-defined timing blocks

    Parameters:
        df: A Pandas datframe of processed Twitter data
        ref: Reference group (string) to be used in title of plot (e.g. relevant or original tweets)
        strat: Which column to use to stratify the timing data (e.g. date or hour_3h)
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1)
    cp = sns.color_palette('hls', 8)

    # Grab any time-filter kwargs
    date_filter = kwargs.get('date_filter', None)
    date_start = kwargs.get('date_start', None)
    date_end = kwargs.get('date_end', None)
    split = kwargs.get('split', None)

    # Filter the data, if provided by user
    if date_filter is True:
        if date_start is not None:
            df = df.loc[df['created_at'] >= date_start]
            if date_end is not None:
                df = df.loc[df['created_at'] <= date_end]
        else:
            if date_end is not None:
                df = df.loc[df['created_at'] <= date_end]

    # Vary error bar widths based on the number of bars to plot
    ewidth = 90/len(df[strat].drop_duplicates().tolist())
    if ewidth > 2:
        ewidth = 2

    # Plot tweet counts and retweet bar plots
    sns.countplot(data=df, x=strat, hue=split, ax=axes[0], palette=cp)
    sns.barplot(data=df, x=strat, hue=split, y='retweet_count', estimator=np.median, errwidth=ewidth, ax=axes[1],
                palette=cp)

    # Format time-axes
    if (strat == 'date') | (strat == 'created_at_date6h') | (strat == 'date_range'):
        rotate = 45
        xlabel = 'Date'
        if strat == 'created_at_date6h':
            xticks = np.arange(-0.5, 59.5, 4)
            xticklabels = ['08-30', '08-31', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '09-07', '09-08',
                           '09-09', '09-10', '09-11', '09-12', '09-13']
            for ax in axes:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, rotation=rotate, ha='right')

            # For 6h-date field, limit y-values to remove effects of any large outliers
            axes[1].set_ylim(0, 50)

    else:
        if strat == 'hour_3h':
            rotate = 45
        else:
            rotate = 0
        xlabel = 'Hour (Eastern Time)'

    # Label and format axes
    labels = ['Count', 'Retweet Count']
    for i, ax in enumerate(axes):
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate, ha='right')
        # ax.tick_params(axis='x', labelrotation=rotate, ha='right')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(labels[i], fontsize=14)

    # Add a title
    fig.suptitle(ref, fontsize=14)

    # Final format and save/show
    plt.tight_layout()
    fig.savefig(ref + '_' + strat + 'by_' + split + '_count_rt.png', dpi=300, bbox_inches='tight')
    plt.show()


def diff_calcs(df, ref, engage_type):
    """
    Calculate source-adjusted/independent diffusion/engagement calculations

    Parameters:
        df: A Pandas dataframe of Twitter data
        ref: A string descriptor of the group of tweets that source baselines are calculated over (e.g. rel, original)
        engage_type: A string descriptor of the engagement metric to calculate for (e.g. retweet, reply, quote, like,
                         all engages)

    Returns:
        The same dataframe as entered, but with several additional columns which contain source-adjusted engagement
            calculations for each tweet

    Outputs:
        None
    """

    # Z-scores
    df.loc[:, ref + '_source_' + engage_type + '_mean'] = df.groupby('user.username')[
        engage_type + '_log_norm'].transform('mean')
    df.loc[:, ref + '_source_' + engage_type + '_sd'] = df.groupby('user.username')[
        engage_type + '_log_norm'].transform('std')
    df.loc[:, ref + '_source_' + engage_type + 'Z'] = (df[engage_type + '_log_norm'] - df[
        ref + '_source_' + engage_type + '_mean']) / df[ref + '_source_' + engage_type + '_sd']

    # Median difference from source median RTs
    df.loc[:, ref + '_source_raw_' + engage_type + '_median'] = df.groupby('user.username')[
        engage_type + '_plus1'].transform('median')
    df.loc[:, ref + '_source_raw_' + engage_type + '_diff'] = df[engage_type + '_plus1'] - df[
        ref + '_source_raw_' + engage_type + '_median']

    # % of median source RTs
    df.loc[:, ref + '_source_raw_' + engage_type + '_per'] = 100 * df[engage_type + '_plus1'] / df[
        ref + '_source_raw_' + engage_type + '_median']

    # Follower-normalized RTs
    df.loc[:, ref + '_follower_' + engage_type] = (10 ** 5) * df[engage_type + '_count'] / df['user.followers_count']

    return df


def media_url_list(df, save=True):
    """
    Pull all media URLs (including preview image URLs for GIFs and videos) into one document, for use in downloading
        images from URLs using wget

    Parameters:
        df: A dataframe of Twitter data, already filtered to only include media tweets
        save: Whether to save the output (to Irma/Images folder in Twitter Analysis)

    Returns:
        A list of URLS arranged in a single column

    Outputs:
        Text file of URLs, if desired
    """
    # Pull out media & preview URLs
    media1_urls = df['media1.url'].drop_duplicates().dropna()
    media2_urls = df['media2.url'].drop_duplicates().dropna()
    media3_urls = df['media3.url'].drop_duplicates().dropna()
    media4_urls = df['media4.url'].drop_duplicates().dropna()
    gif_video_urls = df['media1.preview_image_url'].drop_duplicates().dropna()

    # Concatnate media & preview URLS together and remove the https:// at the beginning
    media_urls = pd.concat([media1_urls, media2_urls, media3_urls, media4_urls, gif_video_urls])
    media_urls = media_urls.str[8:]

    if save is True:
        media_urls.to_csv('Irma\\Images\\irma_media.txt', index=False, header=False)

    return media_urls
