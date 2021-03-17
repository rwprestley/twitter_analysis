import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import rgb2hex
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np
import math
from scipy import stats
from scipy.stats import gaussian_kde
import json
import os
import webbrowser
import plotly.graph_objects as go
import seaborn as sns

# <editor-fold desc="Data merging and filtering">


def media_type_convert(df):
    """
    Converts media-type columns to a consistent standard

    Parameters:
        df: A Pandas dataframe of Twitter data
    """

    # For missing data or JSON based dataframes, a single media-type column already exists. Modify it to distinguish
    # multiple image tweets from single image tweets, simplify the 'gif' title, and rename the column for
    # consistency with other column names.
    if ('media-media_type' in df.columns) is True:
        df.loc[
            (df['media-media_type'] == 'photo') & (
                    df['media-url_num'] == 1), 'media-media_type'] = 'single_photo'
        df.loc[
            (df['media-media_type'] == 'photo') & (df['media-url_num'] > 1),
            'media-media_type'] = 'multi_photo'
        df.loc[df['media-media_type'] == 'animated_gif', 'media-media_type'] = 'gif'
        df.rename(columns={'media-media_type': 'media-type'}, inplace=True)

    # For HIMN CSV-based dataframes, a single media-type column needs to be created.
    else:
        # Convert columns from bool or object to int64.
        media_types = ['media-gif', 'media-multi_photo', 'media-single_photo', 'media-video']
        for type_col in media_types:
            df[type_col] = df[type_col].astype('int64')

        # Create media type column by assessing the max value of the tweet type columns for each row and returning
        # the column name. Remove 'media-' header to leave just the media type.
        df['media-type'] = df[media_types].idxmax(axis=1)
        df['media-type'] = df['media-type'].str[6:]

        # Remove individual media type columns.
        df.drop(columns=media_types, inplace=True)
        df.drop(columns='media-category', inplace=True)

    return df


def media_url_convert(df):
    """
    Reformats media URL column and splits in to seperate columns for each URL

    Parameters:
        df: A Pandas dataframe of Twitter data
    """

    # The media URL column is stored differently based on where the data originates from. Loop through both options
    # to ensure the URL columns are converted regardless of how they are stored.
    media_url_cols = ['media-media_urls', 'tweet-media_urls']
    for url_col in media_url_cols:
        # If the column is in the dataframe, go ahead with conversion then end code. Else, move to the next column.
        if (url_col in df.columns) is True:

            # Format the media URL column, if not already formatted.
            if df[url_col].iloc[0][:1] != 'h':
                # Convert the column to a string and remove leading/trailing brackets.
                df[url_col] = df[url_col].astype(str)
                df[url_col] = df[url_col].str[1:-1]

                # Remove hyphens and extra strings in the string, leaving just a list of plain text URLS seperated
                # only by commas.
                replace_pattern = '|'.join(["'", " "])
                df[url_col] = df[url_col].str.replace(replace_pattern, '')

            # Use the number of commas in the string to count the number of URLS.
            df['media-url_num'] = df[url_col].str.count(",") + 1

            # Create a list of four potential new media columns (there can be a maxiumum of four media attachments
            # to a tweet).
            media_cols = []
            for z in range(1, 5):
                media_cols.append('media-url' + str(z))

            # Calculate the maximum number of media attachments of any tweet in the dataset, and use this to create
            # the correct number of new media URL columns (e.g. a dataset with a max of 2 media attachments should
            # only have two media URL columns - media-url1 and media-url2.
            for z in range(1, 5):
                if z == df['media-url_num'].max():
                    media_cols_max = media_cols[0:z]
                    df[media_cols_max] = df[url_col].str.split(',', expand=True)
                    break
                else:
                    continue

            # Rename the original media URL column.
            df.rename(columns={url_col: 'media-urls'}, inplace=True)

            if url_col == 'tweet-media_urls':
                df.drop('tweet-num_media_urls', axis=1, inplace=True)

            break

        else:
            continue

    return df


def tweet_type_convert(df):
    """
    Converts tweet type columns to a single tweet type column

    Parameters:
        df: A Pandas dataframe of Twitter data
    """

    # Select columns which begin with 'tweet_types' and append to list. Convert columns from bool or object to int64
    tweet_types_cols = []
    for col in df.columns:
        if col[:11] == 'tweet_types':
            tweet_types_cols.append(col)
            df[col] = df[col].fillna(0)
            df[col] = df[col].astype('int64')

    # Create tweet type column by assessing the max value of the tweet type columns for each row and returning the
    # column name. Remove 'tweet_type.' header to leave just the tweet type.
    df['tweet-type'] = df[tweet_types_cols].idxmax(axis=1)
    df['tweet-type'] = df['tweet-type'].str[12:]

    # Remove individual tweet type columns.
    df.drop(columns=tweet_types_cols, inplace=True)

    return df


def merge_orig_codes(df, orig_file):
    """
    Merge tweet data with originator data to include coded originator data

    Parameters:
        df: A tweet database (Pandas dataframe)
        orig_file: Filename/location of originators, which includes scope, affiliation, and agency codes for each
                       originator (string)
    """

    final_originator_codes = pd.read_csv(orig_file)
    final_originator_codes.rename(
        columns={'Originator': 'user-screen_name', 'Scope': 'user-scope', 'Agency': 'user-agency',
                 'Affiliation': 'user-affiliation'}, inplace=True)

    final_originator_codes['user-screen_name'] = final_originator_codes['user-screen_name'].str.lower()
    df['user-screen_name'] = df['user-screen_name'].str.lower()

    df = pd.merge(df, final_originator_codes.iloc[:, 0:4], on='user-screen_name', how='left')

    return df


def himn_convert(file):
    """
    Convert an original HIMN datafile to a dataframe, compatible for merging with other datasets

    Parameters:
        file: Filename/location of the HIMN datafile (string)
    """

    # Read in data
    himn_df = pd.read_csv(file, low_memory=False, header=0, encoding="ISO-8859-1")

    # Remove extraneous columns
    drop_cols = ['concat_key', 'concat_key.1', 'rel_X_fore_Harvey', 'threat_sum', 'type_sum', 'type_WW_RS',
                 'type_WW_text', 'time_Irma', 'rel_Irma', 'fore_Irma', 'time_Harvey']
    diff_cols = [col for col in himn_df.columns if col[:9] == 'diffusion']
    threat_cols = [col for col in himn_df.columns if col[:6] == 'threat']
    himn_df.drop(columns=drop_cols + diff_cols + threat_cols, errors='ignore', inplace=True)

    # Manually change image name types to change source to branding and replace second underscores with hyphens. Also
    # change diffusion-retweet column to diffusion-rt.
    old_cols = ['source_off', 'source_non-off', 'type_key_msg', 'type_threat_impact', 'type_conv_out',
                'type_meso_disc', 'type_rain_fore', 'type_rain_out', 'type_riv_flood', 'type_other_fore',
                'type_other_non-fore', 'threat_trop_gen', 'threat_rain_flood', 'diffusion-retweet_count', 'rel_Harvey',
                'fore_Harvey']
    new_cols = ['branding_off', 'branding_unoff', 'type_key-msg', 'type_threat-impact',
                'type_conv-out', 'type_meso-disc', 'type_rain-fore', 'type_rain-out', 'type_riv-flood',
                'type_other-fore', 'type_other-non-fore', 'threat_trop-gen', 'threat_rain-flood',
                'diffusion-rt_count', 'filter-relevant', 'filter-forecast']
    himn_df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)

    # Add or adjust column prefixes for columns
    user_cols = ['agency', 'affiliation', 'scope']
    new_user_cols = ['user-' + col for col in user_cols]

    tweet_user_cols = [col for col in himn_df.columns if col[:11] == 'tweet-user_']
    new_tweet_user_cols = ['user-' + col[11:] for col in tweet_user_cols]

    image_cols = [col for col in himn_df.columns if col[:4] == 'bran' or col[:4] == 'lang' or col[:4] == 'type' or
                  col[:4] == 'thre']
    new_image_cols = ['image-' + col for col in image_cols]

    # Rename columns
    old_cols = user_cols + tweet_user_cols + image_cols
    new_cols = new_user_cols + new_tweet_user_cols + new_image_cols
    new_cols = [col.lower() for col in new_cols]
    himn_df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)

    # Create truncated tweet id column
    himn_df['tweet-id_trunc'] = himn_df['tweet-id'].astype(str).str[:15]

    # Create/edit media type and URL columns
    himn_df = media_url_convert(himn_df)
    himn_df = media_type_convert(himn_df)

    # Convert created_at column to format that matches missing and JSON datasets
    himn_df['tweet-created_at'] = pd.to_datetime(himn_df['tweet-created_at'], infer_datetime_format=True)
    himn_df['tweet-created_at'] = himn_df['tweet-created_at'].dt.tz_localize('UTC')
    himn_df['tweet-created_at'] = himn_df['tweet-created_at'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

    # Create risk image filter column
    himn_df['filter-risk'] = 1

    # Format forecast and relevance filter codes to be either 1, 0, or nan.
    himn_df.loc[(himn_df['filter-relevant'] != 1) & (himn_df['filter-relevant'] != '1'), 'filter-relevant'] = 0
    himn_df.loc[himn_df['filter-relevant'] == '1', 'filter-relevant'] = 1
    himn_df.loc[(himn_df['filter-forecast'] != 1) & (himn_df['filter-forecast'] != '1'), 'filter-forecast'] = 0
    himn_df.loc[himn_df['filter-forecast'] == '1', 'filter-forecast'] = 1

    return himn_df


def json_convert(file):
    """
    Convert a JSON tweet file to a dataframe of tweet-ids and truncated ids, compatible for merging with HIMN data

    Parameters:
        file: Filename/location for JSON tweet file (string)
    """

    # Read in JSON data
    with open(file, 'r') as f:
        json_data = json.load(f)
    json_data_df = pd.json_normalize(json_data)

    # Modify tweet-id column and create tweet-id trunc column
    json_data_df.rename(columns={'id': 'tweet-id'}, inplace=True)
    json_data_df['tweet-id_trunc'] = json_data_df['tweet-id'].astype(str).str[:15]

    # Remove all non-id columns
    for col in json_data_df.columns:
        if col not in ['tweet-id', 'tweet-id_trunc']:
            json_data_df.drop(columns=col, inplace=True)

    return json_data_df


def old_missing_convert(all_file, fore_file, orig_file):
    """
    Merges two datafiles of new tweet data uncovered in summer 2020, one with filter coding (e.g. risk, relevance,
        and forecast) and one with full coded images, to a dataframe, compatible for merging with other tweet data

    Parameters:
        all_file: Filename/location of all missing tweet data uncovered in summer 2020, coded for filtering
                      criteria (string)
        fore_file: Filename/location of forecast missing tweet data uncovered in summer 2020, fully coded for image
                       branding, type, language, and threat
        orig_file: Filename/location of originators, which includes scope, affiliation, and agency codes for each
                       originator (string)
    """

    # Read in both old missing datafiles and merge together
    old_missing = pd.read_csv(all_file)
    old_missing_fore = pd.read_csv(fore_file)
    old_missing_df = pd.merge(old_missing[['id', 'risk image', 'relevant', 'forecast', 'coded']], old_missing_fore,
                              how='left', on='id')

    # Filter to only include newly discovered tweets
    old_missing_df = old_missing_df.loc[old_missing_df['coded'] == False]

    # Remove extraneous columns
    threat_cols = [col for col in old_missing_df.columns if col[:6] == 'threat']
    drop_cols = \
        ['geolocation', 'user.account_created_at.$date', 'favorites_count', 'user.description', 'coded'] + threat_cols
    old_missing_df.drop(columns=drop_cols, inplace=True)

    for col in old_missing_df.columns:
        if col[:7] == 'Unnamed':
            old_missing_df.drop(col, axis=1, inplace=True)

    # Rename user columns.
    old_missing_df.columns = old_missing_df.columns.str.replace('.', '-')

    # Add column prefixes to image code columns.
    image_cols = [col for col in old_missing_df.columns if col[:4] == 'type'] + ['lang_spanish']
    new_cols = [('image-' + col).lower() for col in image_cols]
    old_missing_df.rename(columns=dict(zip(image_cols, new_cols)), inplace=True)

    # Rename other columns
    old_missing_df.rename(columns={'url_count': 'media-url_num', 'source_off': 'image-branding_off',
                               'source_unoff': 'image-branding_unoff', 'created_at-$date': 'tweet-created_at',
                               'id': 'tweet-id', 'image-type_ww_meso-disc': 'image-type_ww_md', 'text': 'tweet-text',
                               'media_url1': 'media-url1', 'media_url2': 'media-url2', 'media_url3': 'media-url3',
                               'media-media_urls': 'media-urls', 'risk image': 'filter-risk',
                               'relevant': 'filter-relevant', 'forecast': 'filter-forecast'}, inplace=True)

    # Create truncated id column.
    old_missing_df['tweet-id_trunc'] = old_missing_df['tweet-id'].astype(str).str[:15]

    # Create/edit media type and tweet type columns.
    old_missing_df = media_type_convert(old_missing_df)
    old_missing_df = tweet_type_convert(old_missing_df)

    # Convert created_at column to format that matches HIMN and JSON datasets
    old_missing_df['tweet-created_at'] = pd.to_datetime(old_missing_df['tweet-created_at'], infer_datetime_format=True)
    old_missing_df['tweet-created_at'] = old_missing_df['tweet-created_at'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

    # Merge missing data with originator data to include coded originator data.
    old_missing_df = merge_orig_codes(old_missing_df, orig_file)

    return old_missing_df


def new_missing_convert(file, orig_file):
    """
    Convert a datafile of new tweet data uncovered in December 2020 to a dataframe, compatabile for merging with other
        tweet data

    Parameters:
        file: Filename/location of missing tweet data uncovered in December 2020 (string)
        orig_file: Filename/location of originators, which includes scope, affiliation, and agency codes for each
                       originator (string)
    """

    # Read in new missing data, discovered in Dec 2020 (created via image coding package).
    new_missing_df = pd.read_csv(file, encoding='UTF-8')

    # Remove extraneous columns
    new_missing_drop_cols = ['geolocation', 'user.account_created_at', 'link', 'image_count', 'id.trunc']
    new_missing_df.drop(columns=new_missing_drop_cols, errors='ignore', inplace=True)

    for col in new_missing_df.columns:
        if col[:7] == 'Unnamed':
            new_missing_df.drop(col, axis=1, inplace=True)

    # Rename missing user columns.
    new_missing_df.columns = new_missing_df.columns.str.replace('.', '-')

    # Add column prefixes and map 'yes'/'no' to 1/0.
    image_cols = ['trop-out', 'cone', 'arrival', 'prob', 'surge', 'key-msg', 'ww', 'threat-impact', 'conv-out',
                  'meso-disc', 'rain-fore', 'rain-out', 'riv-flood', 'spag', 'text-img', 'model', 'evac',
                  'other-fore', 'other-non-fore', 'video']
    ww_cols = ['ww_exp', 'ww_cone', 'ww_md']
    other_cols = ['official', 'unofficial', 'spanish', 'hrisk_img', 'forecast', 'relevant']

    new_cols = []
    for col in new_missing_df.columns:
        if col in image_cols:
            new_cols.append('image-type_' + col)
            new_missing_df[col] = new_missing_df[col].map({'yes': 1, 'no': 0})
        if col in ww_cols:
            new_cols.append('image-type_' + col)
            new_missing_df[col] = new_missing_df[col].map({'yes': 1, 'no': 0})
        if col in other_cols:
            new_missing_df[col] = new_missing_df[col].map({'yes': 1, 'no': 0})

    new_missing_df.rename(columns=dict(zip(image_cols + ww_cols, new_cols)), inplace=True)

    # Rename other columns
    new_missing_df.rename(columns={'image-type_text-img': 'image-type_text', 'official': 'image-branding_off',
                               'unofficial': 'image-branding_unoff', 'spanish': 'image-lang_spanish',
                               'forecast': 'filter-forecast', 'relevant': 'filter-relevant',
                               'tweet_type': 'tweet-type', 'text': 'tweet-text', 'created_at': 'tweet-created_at',
                               'id': 'tweet-id', 'hrisk_img': 'filter-risk'}, inplace=True)

    # Create truncated id column.
    new_missing_df['tweet-id_trunc'] = new_missing_df['tweet-id'].astype(str).str[:15]

    # Create and/or edit media URL and media type columns.
    new_missing_df = media_url_convert(new_missing_df)
    new_missing_df = media_type_convert(new_missing_df)

    # Convert created_at column to format that matches HIMN and JSON datasets
    new_missing_df['tweet-created_at'] = pd.to_datetime(new_missing_df['tweet-created_at'], infer_datetime_format=True)
    new_missing_df['tweet-created_at'] = new_missing_df['tweet-created_at'].dt.strftime('%Y-%m-%d %H:%M:%S%z')

    # Merge missing data with originator data to include coded originator data.
    new_missing_df = merge_orig_codes(new_missing_df, orig_file)

    return new_missing_df


def merge(him_all_file, himn_coded_file, old_missing_all_file, old_missing_fore_file, new_missing_file, orig_file,
          start, end):
    """
    Merges multiple sources of Twitter data to create a unified dataset

    Parameters:
        him_all_file: Filename/location of CSV datafile containing all HIM tweet data (N = 116k) (string)
        himn_coded_file: Filename/location of CSV datafile with coded data for original risk-image dataset (N = 16k)
                             (string)
        old_missing_all_file: Filename/location of CSV datafile with all data collected during summer 2020 after
                                  noticing lack of tweets in Aug 26 evening timeframe (N ~ 500). Includes codes for
                                  filtering criteria (string)
        old_missing_fore_file: Filename/location of CSV datafile with fully coded forecast data for old missing data
                                   (N = 113) (string)
        new_missing_file: Filename/location of CSV datafile with fully coded data for new/missing data, obtained from
                              HIM all file obtained December 2020 (N ~5000). Includes filtering codes and image codes
                              (string)
        orig_file: Filename/location of CSV datafile with coded originators/sources (string)
        start: Storm start time for time-filtering (tz-aware (UTC) datetime object)
        end: Storm end time for time-filtering (tz-aware (UTC) datetime object)
    """

    # Reorganize and recalculate datasets to include the calculated properties and column names (excluding diffusion
    # calculations)
    him_116k = new_missing_convert(him_all_file, orig_file)
    himn_16k = himn_convert(himn_coded_file)
    old_missing = old_missing_convert(old_missing_all_file, old_missing_fore_file, orig_file)
    new_missing = new_missing_convert(new_missing_file, orig_file)

    # Concatenate missing datasets with HIMN (16k) data to obtain a full set of coded data.
    coded = pd.concat([himn_16k, old_missing, new_missing], join='outer')

    # Merge coded data with 116k HIM dataset
    tweets_harvey_all = pd.merge(him_116k, coded, on='tweet-id_trunc', how='left', suffixes=('', '_y'))
    tweets_harvey_all.drop(tweets_harvey_all.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

    # Create a code for video based on the media type column.
    tweets_harvey_all['image-type_video'] = [1 if x == 'video' else 0 for x in tweets_harvey_all['media-type']]

    # Remove overlaps when video is coded.
    for col in tweets_harvey_all.columns:
        if (col[:10] == 'image-type') & (col != 'image-type_video'):
            tweets_harvey_all.loc[tweets_harvey_all['image-type_video'] == 1, col] = 0

    # Create an English language column to complement the Spanish language column.
    tweets_harvey_all.loc[tweets_harvey_all['image-lang_spanish'] == 1, 'image-lang_english'] = 0
    tweets_harvey_all.loc[tweets_harvey_all['image-lang_spanish'] == 0, 'image-lang_english'] = 1

    # Create a tweet URL column by splitting the last 23 digits from the tweet text column.
    tweets_harvey_all['tweet-url'] = tweets_harvey_all['tweet-text'].str.slice(start=-23)
    tweets_harvey_all['tweet-text'] = tweets_harvey_all['tweet-text'].str.slice(stop=-23)

    # Create time-filter column, using provided start and end dates
    tweets_harvey_all['tweet-created_at'] = pd.to_datetime(tweets_harvey_all['tweet-created_at'],
                                                           format='%Y-%m-%d %H:%M:%S%z')
    tweets_harvey_all.loc[(tweets_harvey_all['tweet-created_at'] >= start) &
                          (tweets_harvey_all['tweet-created_at'] <= end), 'filter-time'] = 1

    # Create source-filter column
    tweets_harvey_all.loc[
        (tweets_harvey_all['user-scope'] == 'Local - Harvey') |
        ((tweets_harvey_all['user-scope'] == 'National/International') &
         ((tweets_harvey_all['user-affiliation'] == 'Gov - Wx - NWS') |
          (tweets_harvey_all['user-affiliation'] == 'Media - Wx'))), 'filter-source'] = 1

    return tweets_harvey_all


def diff_calc_basic(df, diff_folder):
    """
    Calculate retweet, quote tweet, combined RT and QT, and reply counts for provided Twitter dataset

    Parameters:
        df: A dataframe of Twitter data (Pandas dataframe)
        diff_folder: Folder location where JSON diffusion data for each forecast tweet are stored (string)
    """

    # Format ids as strings and make index of dataframe.
    df.reset_index(inplace=True)
    df['tweet-id'] = df['tweet-id'].astype(str)
    df.set_index('tweet-id', inplace=True)

    # Initialize diffusion metrics
    df['diffusion-rt_count'] = ''
    df['diffusion-qt_count'] = ''
    df['diffusion-reply_count'] = ''
    n = 0

    # Calculate diffusion for each tweet in diffusion folder.
    for filename in os.listdir(diff_folder):
        if (filename[:18] in df.index) is True:
            with open(diff_folder + '\\' + filename, 'r') as f:
                data = json.load(f)

                if len(data) != 0:
                    # Convert data for each tweet-id in to a DataFrame (but only if tweet has any diffusion).
                    data_df = pd.json_normalize(data)

                    # Calculate diffusion counts for RT, QT, and replies.
                    df.loc[filename[:18], 'diffusion-rt_count'] = len(data_df.loc[data_df['tweet_types.retweet'] != 0])
                    df.loc[filename[:18], 'diffusion-qt_count'] = \
                        len(data_df.loc[data_df['tweet_types.quote_tweet'] != 0])
                    df.loc[filename[:18], 'diffusion-reply_count'] = len(data_df.loc[data_df['tweet_types.reply'] != 0])

                else:
                    df.loc[filename[:18], 'diffusion-rt_count'] = 0
                    df.loc[filename[:18], 'diffusion-qt_count'] = 0
                    df.loc[filename[:18], 'diffusion-reply_count'] = 0

        # Track progress
        n += 1
        print(str(n) + '/' + str(len(os.listdir(diff_folder))))

    # Calculate combined RT/QT count
    df['diffusion-combined_rt_qt_count'] = df['diffusion-rt_count'] + df['diffusion-qt_count']

    return df


def tweet_diffusion_calc(tweet_df, data_folder, diff_folder):
    # Calculates counts and rates for each diffusion metric (retweet, reply, and quote tweet) for several different
    # timeframes after the posting of the tweet. These calculated metrics are appended to a tweet dataframe as new
    # columns. Required input: a dataframe with data for each tweet, a data folder where data is stored and saved, a
    # diffusion folder within the data folder that contains diffusion data for each tweet as a seperate file, the column
    # order that the calculated tweet_df should be ordered by, and a name to save the calculated tweet_df as
    # (default is "tweets_harvey_calc").

    print('creating columns...')
    # Define lists to be iterated through later.
    rate_times = [5, 10, 15, 30, 60, 120, 240, 360]
    diff_metrics = ['retweet', 'quote_tweet', 'reply']

    # Format ids as strings and make index of dataframe.
    tweet_df.reset_index(inplace=True)
    tweet_df['tweet-id'] = tweet_df['tweet-id'].astype(str)
    tweet_df.set_index('tweet-id', inplace=True)

    # Read diffusion data for each tweet-id in diffusion files folder that matches an id in the tweet dataframe
    # (this prevents reading in outliers and experimental watch/warning images if they have been removed
    # previously).
    n = 0
    for filename in os.listdir(data_folder + '\\' + diff_folder):
        if (filename[:18] in tweet_df.index) is True:
            with open(data_folder + '\\' + diff_folder + '\\' + filename, 'r') as f:
                data = json.load(f)

            # Convert data for each tweet-id in to a DataFrame (but only if tweet has any diffusion).
            if len(data) != 0:
                data_df = pd.json_normalize(data)

                # Rename created-at column.
                data_df.rename(columns={'created_at.$date': 'timestamp'}, inplace=True)
                data_df = tweet_type_convert(data_df)

                # Convert timestamp to a datetime object.
                data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                data_df['timestamp'] = data_df['timestamp'].dt.tz_convert('US/Central')
                data_df.sort_values('timestamp', inplace=True)

                # Obtain the tweet created-at time from tweet-data, matching tweet-data id to the current filename
                # iteration (excluding the .json).
                created_at = pd.to_datetime(tweet_df.loc[tweet_df.index == filename[:18], 'tweet-created_at'].iloc[0]).\
                    tz_convert('US/Central')

                # Calculate the time delta between the tweet's creation and the time diffusion occurred.
                # Convert to minutes.
                data_df['delta'] = data_df['timestamp'] - created_at
                data_df['delta'] = (data_df['delta'].dt.days * 1440) + (data_df['delta'].dt.seconds / 60)

                # For each metric and calculation timeframe, calculate the diffusion count within the timeframe and
                # the rate (in RT/reply/QT per hour). Append value to the tweet dataframe by locating the row where
                # the index is equal to the truncated filename. REQUIRES INDEX TO BE THE TWEET-ID.
                for metric in diff_metrics:
                    for time in rate_times:
                        tweet_df.loc[filename[:18], 'diffusion-' + metric + '_count_' + str(time) + 'm'] = len(
                            data_df.loc[(data_df['tweet-type'] == metric) & (data_df['delta'] <= time)])
                        #tweet_df.loc[filename[:18], 'diffusion-' + metric + '_rate_' + str(time) + 'm'] = len(
                        #    data_df.loc[(data_df['tweet-type'] == metric) &
                        #                (data_df['delta'] <= time)]) * (60 / time)

            # If tweet has no diffusion, set metrics equal to zero.
            else:
                for metric in diff_metrics:
                    for time in rate_times:
                        tweet_df.loc[filename[:18], 'diffusion-' + metric + '_count_' + str(time) + 'm'] = 0
                        #tweet_df.loc[filename[:18], 'diffusion-' + metric + '_rate_' + str(time) + 'm'] = 0

        else:
            print('filename not matched')
            # If tweet has no diffusion, set metrics equal to zero.
            for metric in diff_metrics:
                for time in rate_times:
                    tweet_df.loc[filename[:18], 'diffusion-' + metric + '_count_' + str(time) + 'm'] = np.nan
                    #tweet_df.loc[filename[:18], 'diffusion-' + metric + '_rate_' + str(time) + 'm'] = np.nan

        n += 1
        print(str(n) + '/' + str(len(os.listdir(data_folder + '\\' + diff_folder))))

    # Add or adjust column prefixes.
    old_cols = []
    new_cols = []
    for col in tweet_df.columns:
        if col[:17] == 'diffusion-retweet':
            old_cols.append(col)
            new_cols.append('diffusion-rt' + col[17:])
        if col[:21] == 'diffusion-quote_tweet':
            old_cols.append(col)
            new_cols.append('diffusion-qt' + col[21:])

    new_cols = [col.lower() for col in new_cols]
    tweet_df.rename(columns=dict(zip(old_cols, new_cols)), inplace=True)

    # Create combined RT/QT stats.
    for time in rate_times:
        tweet_df['diffusion-combined_rt_qt_count_' + str(time) + 'm'] = tweet_df['diffusion-rt_count_' + str(
            time) + 'm'] + tweet_df['diffusion-qt_count_' + str(time) + 'm']
        #tweet_df['diffusion-combined_rt_qt_rate_' + str(time) + 'm'] = tweet_df['diffusion-rt_rate_' + str(
        #    time) + 'm'] + tweet_df['diffusion-qt_rate_' + str(time) + 'm']

    # Reset the index
    tweet_df = tweet_df.reset_index()
    #tweet_df.set_index('tweet-id', inplace=True)

    # Save the dataframe as a CSV and JSON file, using user-provided name.
    #tweet_df.to_csv(data_folder + '\\' + tweet_df_name + '.csv')
    #tweet_df.to_json(data_folder + '\\' + tweet_df_name + '.json')

    return tweet_df


def scope_aff_filter(df, col_order, sep_exp=False):
    """
    Renames and reorganizes tweet sources in a tweet dataframe

    Parameters:
        df: A tweet dataframe, where sources are coded (Pandas dataframe)
        col_order: Order in which columns should be organized (list)
        sep_exp: Whether to create seperate source categories for experimental watch/warning images posted by local and
                     national NWS sources (boolean)
    """

    # Rename scope values for clarity.
    df['user-scope'] = df['user-scope'].str[:5]
    scope_merge = {'Local': 'Local', 'Natio': 'National'}
    df['user-scope'].replace(to_replace=scope_merge, inplace=True)

    # Rename affiliation values for clarity.
    replace_dict = {'Gov - Wx - NWS': 'NWS', 'Media - Wx': 'Wx Media', 'Media - News': 'News Media',
                    'Gov - Wx - Non-NWS': 'Non-NWS Wx Gov', 'Gov - EM': 'EM', 'Gov - Other': 'Other Gov',
                    'Other - Wx': 'Other Wx', 'Other - Non-Wx': 'Other Non-Wx'}
    df['user-affiliation'].replace(replace_dict, inplace=True)

    # Create binary columns for local and national tweets.
    df.loc[df['user-scope'] == 'Local', 'user-scope_loc'] = 1
    df.loc[df['user-scope'] == 'National', 'user-scope_nat'] = 1

    # Create binary columns for individual and organizational tweets.
    df.loc[df['user-agency'] == 'Individual', 'user-agency_ind'] = 1
    df.loc[df['user-agency'] == 'Organization', 'user-agency_org'] = 1

    # Merge other-wx and other-nonwx in to general other category.
    df.loc[(df['user-affiliation'] == 'Other Wx') | (df['user-affiliation'] == 'Other Non-Wx'),
           'user-affiliation'] = 'Other'

    if sep_exp is True:
        # Create an affiliation category for NWS experimental accounts.
        df.loc[(df['user-affiliation'] == 'NWS') & (df['image-type_ww_exp'] == 1),
               'user-affiliation'] = 'NWS (Exp)'

    if len(df.loc[df['user-affiliation'] == 'Wx Bloggers']) == 0:
        # Rename other to bloggers, but only for local originators.
        df.loc[
            (df['user-affiliation'] == 'Other') & (df['user-scope'] == 'Local'), 'user-affiliation'] = 'Wx Bloggers'

    # Concatenate scope and affiliation columns.
    df['user-scope_aff'] = df['user-scope'] + str(' ') + df['user-affiliation']
    unique_scope_affs = df['user-scope_aff'].unique()

    # Merge all scope-affiliation combinations with less than 100 tweets total in to Other. This number is arbitrary.
    for sa in unique_scope_affs:
        if len(df.loc[df['user-scope_aff'] == sa]) < 100:
            df.loc[df['user-scope_aff'] == sa, 'user-affiliation'] = 'Non-NWS Government'

    # Re-concatenate scope and affiliation columns.
    df['user-scope_aff'] = df['user-scope'] + str(' ') + df['user-affiliation']

    # Import column order CSV and reorder the final dataset.
    df = df.reset_index().reindex(columns=col_order)
    df.set_index('tweet-id', inplace=True)

    return df


def image_filter(df):
    """
    Renames and reorganizes image categorizations in a coded tweet dataframe

    Parameters:
        df: A dataframe of coded Twitter data (Pandas dataframe)
    """

    # Define final list of image columns
    image_cols = ['image-type_multi', 'image-type_other-non-fore', 'image-type_other-fore', 'image-type_key-msg',
                  'image-type_model', 'image-type_riv-flood', 'image-type_conv', 'image-type_rain',
                  'image-type_cone', 'image-type_text', 'image-type_trop-out', 'image-type_ww_exp', 'image-type_ww']

    # Remove experimental watch/warning graphics from watch/warning code.
    df.loc[(df['image-type_ww_exp'] == 1) | (df['image-type_ww_exp'] == '1'), 'image-type_ww'] = 0

    # Remove watch/warning overlaps with cone and mesoscale discussion.
    df.loc[(df['image-type_ww_cone'] == 1) | (df['image-type_ww_md'] == 1), 'image-type_ww'] = 0

    # Remove evac/preparedness overlaps with text, where there is only one media URL
    df.loc[(df['image-type_evac'] == 1) & (df['image-type_text'] == 1) & (df['media-url_num'] == 1),
           'image-type_evac'] = 0

    # Where other-non-forecast overlaps with text, change text code to Other - Forecast.
    df.loc[(df['image-type_other-non-fore'] == 1) & (df['image-type_text'] == 1), 'image-type_other-fore'] = 1
    df.loc[(df['image-type_other-non-fore'] == 1) & (df['image-type_text'] == 1), 'image-type_text'] = 0

    # Change text code to other - forecast for specific tweets (infographics where text overlays were coded as text, and
    # newspaper front pages)
    change_ids = [901912382661943296, 901973700718755842, 902037161293344768, 902229526213787648, 902381636863569920,
                  902440415680397312, 902776264024580097, 901701229520326656, 901875772159295488, 901990595694141440,
                  902137209779888128]
    df.loc[df.index.isin(change_ids), 'image-type_text'] = 0
    df.loc[df.index.isin(change_ids), 'image-type_other-fore'] = 1

    # Merge rainfall forecast, rainfall outlook, and WPC mesoscale discussions.
    df['image-type_rain'] = df['image-type_rain-fore'] + df['image-type_rain-out'] + df['image-type_meso-disc_wpc']
    df.loc[df['image-type_rain'] > 1, 'image-type_rain'] = 1

    # Merge convective outlook and SPC mesoscale discussion.
    df['image-type_conv'] = df['image-type_conv-out'] + df['image-type_meso-disc_spc']
    df.loc[df['image-type_conv'] > 1, 'image-type_conv'] = 1

    # Return split-off categories to other-forecast (except for model output and text).
    df['image-type_other-fore'] = df['image-type_evac'] + df['image-type_other-fore']
    df.loc[df['image-type_other-fore'] > 1, 'image-type_other-fore'] = 1

    # Merge model output with spaghetti plots.
    df['image-type_model'] = df['image-type_model'] + df['image-type_spag']
    df.loc[df['image-type_model'] > 1, 'image-type_model'] = 1

    # Remove overlaps when key messages is coded.
    for col in df.columns:
        if (col[:10] == 'image-type') & ((col != 'image-type_key-msg') & (col != 'image-type_sum')):
            df.loc[df['image-type_key-msg'] == 1, col] = 0

    # Merge image types with small counts with other-forecast.
    df['image-type_other-fore'] = df['image-type_surge'] + df[
        'image-type_threat-impact'] + df['image-type_prob'] + df['image-type_arrival'] + df['image-type_other-fore'] + \
        df['image-type_video']
    df.loc[df['image-type_other-fore'] > 1, 'image-type_other-fore'] = 1

    # Remove other-non-forecast overlaps with other forecast content where there is only one media URL
    for col in image_cols[2:]:
        df.loc[(df['image-type_other-non-fore'] == 1) & (df['media-url_num'] == 1) & (df[col] == 1),
               'image-type_other-non-fore'] = 0

    # Make type_sum dynamic and responsive to changes in coding.
    df['image-type_sum'] = df[image_cols[1:]].sum(axis=1)

    # Create a multi-code category
    df.loc[df['image-type_sum'] > 1, 'image-type_multi'] = 1

    # Create image type column by assessing the max value of the image type columns for each row and returning the
    # column name (except for Multiple, which is assigned for all tweets with more than one image type)
    df['image-type'] = df[image_cols[1:]].idxmax(axis=1)
    df.loc[df['image-type_sum'] > 1, 'image-type'] = 'Multiple'

    # Format image type column to have more descriptive/complete names.
    image_types = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
                   'River Flood Forecast', 'Convective Outlook/Forecast', 'Rainfall Outlook/Forecast', 'Cone', 'Text',
                   'Tropical Outlook', 'Watch/Warning (Exp)', 'Watch/Warning']
    df['image-type'].replace(dict(zip(image_cols, image_types)), inplace=True)

    return df


# This function is used as part of the filtering process but can also be useful for narrowing the final dataset further.
def tweet_filter(tweet_df, **kwargs):
    """
    Filter a tweet dataframe based on user-provided criteria

    Parameters:
        tweet_df: A Pandas dataframe of Twitter data

    Keyword Arguments:
        filters: A list of filtering steps to commit (choose among 'time', 'source', 'risk', 'relevant', and 'forecast')
        rt_range: A range of values between lower and upper bounds for retweet values (useful for filtering outliers)
        reply_range: A range of values between lower and upper bounds for reply values
        image_range: A list of images to include in the filtered dataset
        source_range: A list of sources/originator groupings to include in the filtered dataset
        date_range: A list with a start and end-time to filter data between
        cols: A list of columns to include in the filtered dataset

    """

    # Read in optional filtering arguments.
    filters = kwargs.get('filters', None)
    rt_range = kwargs.get('rt_range', None)
    reply_range = kwargs.get('reply_range', None)
    image_range = kwargs.get('image_range', None)
    source_range = kwargs.get('source_range', None)
    date_range = kwargs.get('date_range', None)
    cols = kwargs.get('cols', None)

    if filters is not None:
        # Filter dataset progressively, based on user-provided list of criteria to filter on
        for f in filters:
            tweet_df = tweet_df[(tweet_df['filter-' + f] == 1) | (tweet_df['filter-' + f] == '1')]

    if rt_range is not None:
        # Filter dataset to only include tweets in a user-given range of retweet values.
        tweet_df = tweet_df[tweet_df['diffusion-rt_count'].isin(rt_range)]

    if reply_range is not None:
        # Filter dataset to only include tweets in a user-given range of reply values.
        tweet_df = tweet_df[tweet_df['diffusion-reply_count'].isin(reply_range)]

    if image_range is not None:
        # Filter dataset to only include tweets that are coded as one of a user-provided list of image types.
        tweet_df = tweet_df[tweet_df['image-type'].isin(image_range)]

    if source_range is not None:
        # Filter dataset to only include tweets that are coded as one of a user-provided list of source types.
        tweet_df = tweet_df[tweet_df['user-scope_aff'].isin(source_range)]

    if date_range is not None:
        # Filter dataset to only include tweets posted inbetween a user-provided set of datetime objects.
        tweet_df = tweet_df[(tweet_df['tweet-created_at'] >= date_range[0]) &
                            (tweet_df['tweet-created_at'] < date_range[1])]

    if cols is not None:
        # Filter dataset to only include given columns.
        tweet_df = tweet_df[cols]

    return tweet_df
# </editor-fold>


# <editor-fold desc="Tables/Data Summaries">
# Table 1
def scope_counts(himn_file, missing_file, show=True, save=False):
    # This function creates two tables. The first provides the number of tweets for each coded user scope for each
    # filtering step, merging the values in the original datafile with missing data. The second table is the same,
    # except that it displays the number of accounts.

    # Required inputs: string datafile location for the HIMN file
    # (e.g. 'harvey_irma_twitter_data.csv') and the missing datafile (e.g. 'tweets_harvey_missing.csv'). User can choose
    # whether to show and/or save the tables.

    # Of note - the missing file input is NOT the same as the missing file input when merging databases. The missing
    # file here needs to include all of the missing input (not just the forecast content) and must include columns
    # with binary codes for risk image, relevant, and forecast content.

    # Define the unique scope values.
    scopes = ['Local - Harvey', 'National/International', 'Local - Irma', 'Local - Not Harvey or Irma', 'Other/Unknown']

    # Input HIMN dataset from csv.
    himn = pd.read_csv(himn_file, low_memory=False, header=0, encoding="ISO-8859-1")
    himn = himn.sort_values('tweet-user_screen_name')

    # Input missing file dataset from csv (must include all data filtering coding).
    missing = pd.read_csv(missing_file, low_memory=False, header=0, encoding="ISO-8859-1")
    missing['user_screen_name'] = missing['user_screen_name'].str.lower()

    # Progressively filter HIMN dataset.
    himn_risk_image = himn
    himn_time = himn_risk_image.loc[himn_risk_image['time_Harvey'] == 1]
    himn_rel = himn_time.loc[himn_time['rel_Harvey'] == '1']
    himn_fore = himn_rel.loc[himn_rel['fore_Harvey'] == '1']
    himn_final = himn_fore.loc[
        (himn_fore['scope'] == 'Local - Harvey') | (himn_fore['scope'] == 'National/International')]
    himn_dfs = [himn_risk_image, himn_time, himn_rel, himn_fore, himn_final]

    # Progressively filter missing dataset.
    missing_risk_image = missing.loc[(missing['coded'] is False) & (missing['risk image'] == 1)]
    missing_time = missing_risk_image
    missing_rel = missing_time.loc[missing_time['relevant'] == 1]
    missing_fore = missing_rel.loc[missing_rel['forecast'] == 1]
    missing_final = missing_fore.loc[
        (missing_fore['scope'] == 'Local - Harvey') | (missing_fore['scope'] == 'National/International')]
    missing_dfs = [missing_risk_image, missing_time, missing_rel, missing_fore, missing_final]

    # Calculate the total number of HIMN and missing tweets for each filtered dataset and each scope value. Sum the
    # HIMN and missing numbers. Format as a dataframe.
    tc_df = pd.DataFrame()
    for i in range(0, len(himn_dfs)):
        tc = himn_dfs[i].groupby(['scope'])['tweet-id'].count().add(missing_dfs[i].groupby(['scope'])['id'].count(),
                                                                    fill_value=0)
        tc_df = pd.concat((tc_df, tc), axis=1)
    tc_df.columns = ['Risk Images', 'Time-filtered', 'Relevance-filtered', 'Forecast-filtered', 'Scope-filtered']
    tc_df = tc_df.reindex(scopes)
    tc_df.loc['Grand Total'] = tc_df.sum()
    tc_df = tc_df.fillna(0)
    tc_df = tc_df.astype(np.int64)

    # Show the table, if desired.
    if show is True:
        print(tc_df)

    # Save the table, if desired.
    if save is True:
        tc_df.to_csv('tweet_counts_scope.csv')

    # Calculate the total number of unique HIMN and missing accounts for each filtered dataset and scope value. Sum the
    # two. Format as a dataframe.
    ac_df = pd.DataFrame()
    ac_tot = None
    for i in range(0, len(himn_dfs)):
        ac_himn = []
        ac_miss = []
        for scope in scopes:

            # Obtain a list of unique authoritative sources for each filtered dataset and scope combination between
            # HIMN and missing datasets.
            acts_himn = himn_dfs[i].loc[himn_dfs[i]['scope'] == scope]['tweet-user_screen_name'].unique().tolist()
            acts_miss = missing_dfs[i].loc[missing_dfs[i]['scope'] == scope]['user_screen_name'].unique().tolist()

            # Count the number of authoritative sources in HIMN dataset for each scope.
            ac_himn.append(len(acts_himn))

            # Compare each element in missing sources list to HIMN sources list. If the element is not in the HIMN list,
            # add one.
            n = 0
            for missing_source in acts_miss:
                if (missing_source in acts_himn) is True:
                    n += 0
                else:
                    n += 1

            # Obtain a count of unique authoritive sources in missing dataset. Add to number calculated for HIMN
            # dataset.
            ac_miss.append(n)
            ac_tot = [sum(x) for x in zip(ac_himn, ac_miss)]

        ac_df[str(i)] = ac_tot
    ac_df.columns = ['Risk Images', 'Time-filtered', 'Relevance-filtered', 'Forecast-filtered', 'Scope-filtered']
    ac_df.index = scopes
    ac_df.loc['Grand Total'] = ac_df.sum()

    # Show the table, if desired.
    if show is True:
        print(ac_df)

    # Save the figure, if desired.
    if save is True:
        ac_df.to_csv('account_counts_scope.csv')


# Table 2/3/4
def descr_stats(df, columns, values, labels, metrics):
    # This function creates a dataframe with descriptive statistics, including the count, the number of accounts, and
    # the median, maximum, and percent of tweets with at least one diffusion metric (user-provided: retweet, quote
    # tweet, or reply; user can provide multiple metrics). The function returns the dataframe, at which point the user
    # can choose to show or save the output outside of the function.

    # Required inputs: a tweet dataframe, a column or list of columns that contains the values of interest, a value or
    # list of values to search for within the column(s), labels for the unique values, and a list of diffusion metrics
    # to calculate mean/max/percent with for. Metrics must be formatted as they are formatted in the column names: 'rt'
    # for retweet, 'qt' for quote tweet, 'reply' for reply.

    # To calculate descriptive statistics for variables for which all the values are stored in one column (e.g. image
    # type --> 'image-type' or user source --> 'user-scope_aff'), only input the one column (as a string in a list,
    # not just a string).

    # Example: to obtain descriptive retweet and reply statistics for image-type, input:
    # import twitter_toolkit as ttk
    # ttk.descr_stats(*df_name*, columns=['image-type'], values=*list of image type*, labels=*list of image types*,
    #                       metrics=['rt', 'reply'])

    # For variables where values are stored in multiple columns (e.g. image language is stored in 'image-lang_spanish'
    # and 'image-lang_english', include both columns of interest. For values, put [1], since you are searching for
    # the instances where the value in the column is equal to one. Include descriptive labels.

    # Example: to obtain descriptive retweet statistics for image language, input:
    # import twitter_toolkit as ttk
    # ttk.descr_stats(*df_name*, columns=['image-lang_spanish', 'image-lang_english'], values=[1],
    #                       labels=['Spanish, 'English'], metrics=['rt'])

    # Calculate tweet and account count for each user-provided combination of column and value.
    count = []
    account_count = []
    for col in columns:
        for val in values:
            count.append(len(df.loc[df[col] == val]))
            account_count.append(len(df.loc[df[col] == val]['user-screen_name'].unique()))

    # Calculate tweet and account count for the entire dataset
    count.append(len(df))
    account_count.append(len(df['user-screen_name'].unique()))

    # Create a dataframe to store the count and account data, along with an index, formed from user-provided descriptive
    # labels.
    descr_dict = {'index': labels + ['All'], 'Accounts': account_count, 'Tweet Count': count}
    df_out = pd.DataFrame(descr_dict)

    # Calculate median, maximum, and percent with values for each user-provided metric and for each user-provided
    # combination of column and value.
    for metric in metrics:
        median_count, max_count, total_count, per_count = ([] for _ in range(4))
        for col in columns:
            for val in values:
                median_count.append(df.loc[df[str(col)] == val]['diffusion-' + metric + '_count'].median())
                max_count.append(df.loc[df[str(col)] == val]['diffusion-' + metric + '_count'].max())
                total_count.append(df.loc[df[str(col)] == val]['diffusion-' + metric + '_count'].sum())
                per_count.append(
                    100 * len(df.loc[(df['diffusion-' + metric + '_count'] > 0) & (df[col] == val)]) /
                    (df.loc[df[col] == val]['diffusion-' + metric + '_count'].count()))

        # Calculate median, maximum and percent with values for the entire dataset for each user-provided metric
        median_count.append(df['diffusion-' + metric + '_count'].median())
        max_count.append(df['diffusion-' + metric + '_count'].max())
        total_count.append(df['diffusion-' + metric + '_count'].sum())
        per_count.append(100 * len(df.loc[df['diffusion-' + metric + '_count'] > 0]) / len(df))

        # Append the median, maximum, and percent with values for each metric.
        df_out['Median ' + metric] = median_count
        df_out['Max ' + metric] = max_count
        df_out['Total ' + metric] = total_count
        df_out['% with ' + metric] = per_count

    # Format the dataframe for display.
    df_out.set_index('index', inplace=True)
    pd.options.display.float_format = '{:.1f}'.format

    # Return the data summary to the user.
    return df_out


# Other tables
def mannwhitneyu_test(df, by, how, metric):
    # This function calculates two-sided Mann-Whitney U p-value comparisons for image types or user sources. The
    # function returns a dataframe of p-values, either as a matrix or as a list, depending on user input.

    # Required inputs: a tweet dataframe (df), a column name (by) to gather unique values from
    # (for images --> 'image-type', for user source --> 'user-scope_aff'), whether the test should be performed as a
    # pooled comparison (median of one group to median of everything else) or a matrix (compare median of one group to
    # the median of each other group individually) (how), and the diffusion metric that serves as the basis of
    # comparison (metric).

    # Note: the metric is not constrained to just final tweet count - any count or rate statistic can be compared. As
    # such, user-input must include the full column name (e.g. 'diffusion-rt_count', 'diffusion-reply_count_360m',
    # 'diffusion-qt_rate_5m').

    # Obtain the unique values in the user-provided column and append as 'items' to a dataframe.
    pval_df = pd.DataFrame()
    items = df[by].unique().tolist()
    pval_df['items'] = items

    # If the user selects a matrix comparison, select all the values in one group and compare the median to the median
    # of values in every other group, individually. Append the p-value result of the two-sided Mann-Whitney U test to
    # the dataframe, one row and one column for each individual group.
    if how == 'matrix':
        for item1 in items:
            items_pval = []
            for item2 in items:
                x = df.loc[df[by] == item1]
                y = df.loc[df[by] == item2]
                # print(source1 + ' vs ' + source2)
                statistic, pvalue = stats.mannwhitneyu(x[metric], y[metric], alternative='two-sided')
                items_pval.append(pvalue)
            pval_df[item1] = items_pval

            # Format to only display four decimal points.
            pd.options.display.float_format = '{:.4f}'.format

    # If the user selects a pooled comparison, select all the values in one group and compare the median to the median
    # of values not in that group. Append the p-value result of the two-sided Mann-Whitney U test to the dataframe, in
    # addition the median and count of the in-group and the out-group.
    elif how == 'pooled':
        group_pval = []
        median_in = []
        median_out = []
        count_in = []
        count_out = []
        for item in items:
            x = df.loc[df[by] == item]
            y = df.loc[df[by] != item]
            median_in.append(x[metric].median())
            median_out.append(y[metric].median())
            count_in.append(len(x))
            count_out.append(len(y))
            statistic, pvalue = stats.mannwhitneyu(x[metric], y[metric], alternative='two-sided')
            group_pval.append(pvalue)

        pval_df['pooled_pval'] = group_pval
        pval_df['median_in'] = median_in
        pval_df['median_out'] = median_out
        pval_df['count_in'] = count_in
        pval_df['count_out'] = count_out

        # Format to only display four decimal points for p-value column.
        pval_df['pooled_pval'] = pval_df['pooled_pval'].map('{:.4f}'.format)

    # If the user does not provide a proper response to "how", return an error.
    else:
        return '"How" not valid. Please choose matrix or pooled. '

    # Format the dataframe for display.
    pval_df.set_index('items', inplace=True)

    # Return the p-value dataframe to the user.
    return pval_df


def timeseries_table(df, freq, show=True, save=False):
    # This function returns a dataframe with the tweet count and retweet sum for each image type (in addition to a sum
    # of watch/warning and non-watch/warning images) for each time-bin of a user-provided frequency (in minutes). User
    # can choose whether to show and/or save data output (default is to show but not save).

    # Notes: Function assumes that input dataframe does not already have 'tweet-created_at' as the index. Frequency must
    # be provided as an integer, in minutes (e.g. for 3h, input 180). If user chooses to save output, note that output
    # is saved to a "Timing" folder. If this folder does not exist, an error will be raised.

    # Set the dataframe index to the created at column.
    df = df.set_index('tweet-created_at')

    # Format frequency as a string (for groupby) and convert to a string with the number of hours for the title.
    freq_str = str(freq) + 'Min'
    freq_title = str('{:0.0f}'.format(freq / 60)) + 'h'

    # Group the dataframe into time bins for each image type and calculate the tweet count.
    time_gb = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).count()['tweet-id_trunc'].unstack()

    # Sum the counts for watch/warning images and non-watch/warning images.
    time_gb['ww_sum'] = time_gb.iloc[:, -2:].sum(axis=1)
    time_gb['nonww_sum'] = time_gb.iloc[:, :-3].sum(axis=1)

    # Set the columns to be suffixed by 'count' (to distinguish from RT sum columns created later).
    new_cols = []
    for col in time_gb.columns:
        new_cols.append(col + '.count')
    time_gb.columns = new_cols

    # Group the dataframe into time bins for eeach image type and calculate the retweet sum.
    rt_time_gb = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).sum()['diffusion-rt_count'].unstack()

    # Sum the retweet sums for watch/warning images and non-watch/warning images.
    rt_time_gb['ww_sum'] = rt_time_gb.iloc[:, -2:].sum(axis=1)
    rt_time_gb['nonww_sum'] = rt_time_gb.iloc[:, :-3].sum(axis=1)

    # Set the columns to be suffixed by 'rt.sum' (to distinguish from count columns created earlier).
    new_cols = []
    for col in rt_time_gb.columns:
        new_cols.append(col + '.rt_sum')
    rt_time_gb.columns = new_cols

    # Concatenate the count and RT sum tables together.
    count_rt_sum = pd.concat([time_gb, rt_time_gb], axis=1)

    # Show the output, if desired.
    if show is True:
        print(count_rt_sum)

    # Save the output, if desired (outputs to a seperate folder).
    if save is True:
        count_rt_sum.to_csv('Timing\\counts_rt_sum_' + freq_title + '.csv')


def crosstab_image_source(df, image_range, show=True, save=False):
    # This function takes a dataframe with pre-calculated RT count data over set time periods (e.g. 5m, 6h) and outputs
    # a cross-tabulation between user-provided image types and all user sources (e.g. Local NWS). Includes a count of
    # the number of tweets, sums of total RT and replies, and median values of RT count at each set time period for
    # each image/source pairing, User can choose to show and/or save the formatted data output (default is to show but
    # not save).

    # Note: if user opts to save the output, it will be saved to a "RT Rates" folder. If this folder does not exist,
    # an error will be returned.

    # Select columns with RT count data (for all time periods and total RT).
    count_cols = []
    for col in df.columns:
        if col[:18] == 'diffusion-rt_count':
            count_cols.append(col)

    # Sort image_range in alphabetical order.
    image_range = sorted(image_range)

    # For each image, select data with that image code.
    crosstab_source_image = pd.DataFrame()
    for image in image_range:
        df_image = df.loc[df['image-type'] == image]

        # Calculate median RT for each set time period, count, RT sum, and reply sum for each source.
        df_image_source = df_image.groupby('user-scope_aff')[count_cols].median()
        df_image_source['count'] = df_image.groupby('user-scope_aff')['tweet-id_trunc'].count().tolist()
        df_image_source['RT Sum'] = df_image.groupby('user-scope_aff')['diffusion-rt_count'].sum().tolist()
        df_image_source['Reply Sum'] = df_image.groupby('user-scope_aff')['diffusion-reply_count'].sum().tolist()

        # Re-order the columns.
        count_cols_extra = count_cols.copy()
        count_cols_extra.insert(0, 'Reply Sum')
        count_cols_extra.insert(0, 'RT Sum')
        count_cols_extra.insert(0, 'count')
        df_image_source = df_image_source[count_cols_extra]

        # Rename the columns.
        new_cols = ['count', 'RT Sum', 'Reply Sum', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '6h', 'Total']
        df_image_source.columns = new_cols

        # Sort the dataframe for each image by count.
        df_image_source.sort_values('count', ascending=False, inplace=True)

        # Make the index the image type.
        df_image_source.reset_index(inplace=True)
        df_image_source.index = [image]*len(df_image_source)

        # Create empty rows between each image type in the dataset.
        df_image_source = df_image_source.append(pd.Series(name=''))

        # Append the data for each image type to each other image type to obtain a full, formatted data summary.
        crosstab_source_image = crosstab_source_image.append(df_image_source)

    # Show the output, if desired.
    if show is True:
        print(crosstab_source_image)

    # Export the dataset, if desired.
    if save is True:
        crosstab_source_image.to_csv('RT Rates\\crosstab_source_image.csv')


def diff_data(df, diff_folder, metric):
    # This function creates a database of all retweets, replies, or quote tweets (based on user input) for every tweet
    # in a user supplied dataframe of tweet data, using data in a user-provided diffusion folder, which includes all
    # diffusion stored as a JSON file for each tweet. The resulting output is exported as a CSV.

    # WARNING: GIVEN THE INTENSIVE LOOPING REQUIRED TO CREATE THIS DATABASE, IT TAKES 10-15 MINUTES TO RUN FOR THE FULL
    # HARVEY DATASET (N=2343).

    # Note: Function assumes that the user-provided df has 'tweet-id' set as index. Metric must be formatted as its
    # formatted in the JSON files (retweet: 'retweet', quote tweet: 'quote tweet', reply: 'reply').

    # Loop through each filename in the user-provided diffusion folder.
    diff_df = pd.DataFrame()
    for filename in os.listdir(diff_folder):

        # Match the filename to the tweet-id in the user-provided tweet dataframe.
        if (filename[:18] in df.index.astype(str)) is True:

            # If a match is found, open the diffusion file.
            with open(diff_folder + '\\' + filename, 'r') as f:
                data = json.load(f)

                # Convert data for each tweet-id in to a DataFrame (but only if tweet has any diffusion).
                if len(data) != 0:
                    data_df = pd.json_normalize(data)

                    # Only include diffusion for the specified metric (retweet, reply, or quote tweet).
                    data_df_reply = data_df.loc[data_df['tweet_types.' + metric] != 0]

                    # Merge diffusion data for each tweet together.
                    diff_df = pd.concat([diff_df, data_df_reply])

    # Merge tweet data information together with diffusion data (after formatting the diffusion data ids).
    diff_df['id2'] = diff_df['id2'].astype(np.int64)
    diff_df.rename(columns={'id2': 'tweet-id'}, inplace=True)
    df = df.merge(diff_df, on='tweet-id', how='outer')

    df.to_csv('all_' + metric + '.csv')


def rt_timeseries_table(df, gb, metric, show=True, save=False):
    # This function calculates the median value for a diffusion metric of the user's choice at each of the
    # pre-calculated time periods after tweet creation (e.g. 5m, 10m, 4h, 6h).

    # Required inputs: A tweet dataframe with pre-calculated count values at set times, a column to group the dataframe
    # by (usually 'image-type' for images or 'user-scope_aff' for sources), and a diffusion metric to calculate median
    # values for. User can choose whether to show/save output (default is to show, not to save).

    # Note: metric must be formatted to match how it is formatted in column names (retweet: 'rt', reply: 'reply', quote
    # tweet: 'qt').

    # Gather all the diffusion count columns for the user-provided metric.
    count_cols = []
    for col in df.columns:
        if ((('diffusion-' + metric) in col) is True) & (('count' in col) is True):
            count_cols.append(col)

    # Calculate the median value for each retweet diffusion column for each image or source grouping.
    df_gb = df.groupby([gb])[count_cols].median()

    # Rename the columns.
    new_cols = ['5m', '10m', '15m', '30m', '1h', '2h', '4h', '6h', 'Total']
    df_gb = df_gb.set_axis(new_cols, axis=1)

    # Show output, if desired.
    if show is True:
        print(df_gb)

    # Save output, if desired.
    if save is True:
        df_gb.to_csv(gb + '_timeseries_' + metric + '.count.csv')

    # Calculate the percentage of the final retweet value for each retweet diffusion column for each image or source
    # grouping. Format to display percentages rounded to the nearest whole number.
    df_gb_per = pd.DataFrame()
    for col in df_gb.columns:
        df_gb_per[col] = round((df_gb[col] / df_gb['Total']) * 100)
    pd.options.display.float_format = '{:.0f}'.format

    # Show output, if desired.
    if show is True:
        print(df_gb_per)

    # Save output, if desired.
    if save is True:
        df_gb_per.to_csv(metric + '_timeseries_per.csv', float_format='%.0f')


def cat_midpoint(df, cat_col, weight_col, show=False):
    """
    Calculates the midpoint and diffusion-weighted midpoint (time) for each unique value in a tweet categorization
        scheme. Returns a sorted (by diffusion-weighted midpoint) list of unique categorizations.

    Parameters:
        df: A tweet dataframe with coded categorizations and diffusion data (Pandas dataframe)
        cat_col: Name of column where categorized data is stored (string)
        weight_col: Name of column where data that midpoint should be weighted by is stored (string)
        show: Whether or not to display the midpoints and weighted midpoint for each unique value, sorted from earliest
                  weighted midpoint to latest (Boolean, default False)
    """

    # Initialize mean and weighted mean variables
    mean = []
    mean_weighted = []

    # For each unique value in the categorization column...
    for item in df[cat_col].drop_duplicates().tolist():

        # Select only tweets coded as the unique value
        item_df = df.loc[df[cat_col] == item]

        # Calculate the mean/midpoint of the selected tweets
        item_df['tweet-created_at'] = item_df['tweet-created_at'].astype(np.int64)
        item_mean = pd.to_datetime(item_df['tweet-created_at'].mean())
        mean.append(item_mean)

        # Calculate the diffusion-weighted mean/midpoint of the selected tweets
        item_df['weighted'] = item_df['tweet-created_at'] * (item_df[weight_col] / item_df[weight_col].sum())
        item_weighted = pd.to_datetime(item_df['weighted'].sum())
        mean_weighted.append(item_weighted)

    # Combine unique values, midpoint, and weighted midpoint in to a dataframe and sort by diffusion-weighted mean
    mp_df = pd.DataFrame({cat_col: df[cat_col].drop_duplicates().tolist(), 'mean': mean,
                          'mean_weighted': mean_weighted})
    mp_df['diff'] = mp_df['mean_weighted'] - mp_df['mean']
    mp_df.sort_values('mean_weighted', inplace=True)

    # Show, if desired
    if show is True:
        print(mp_df)

    return mp_df[cat_col].tolist()


# </editor-fold>

# <editor-fold desc="Figures">


# Figure 1: sankey diagram (re-code)
def sankey(df):
    """
    Produces a Sankey plot visualizing the filtering of the tweet data

    Parameters:
        df: A Pandas dataframe of Twitter data
    """

    # Initialize filtering data
    df_filter = df.copy()
    filters = ['time', 'source', 'risk', 'relevant', 'forecast']
    f_tweet = []
    f_source = []

    # Create tweet and source counts for each filtering step, for use in Sankey diagram
    for i in np.arange(0, len(filters) - 1):
        df_filter = df_filter.loc[df_filter['filter-' + filters[i]] == 1]
        df_filter2 = df_filter.loc[df_filter['filter-' + filters[i+1]] == 1]
        f_tweet.append(len(df_filter2))
        f_tweet.append(len(df_filter) - len(df_filter2))
        f_source.append(len(df_filter2['user-screen_name'].drop_duplicates().tolist()))
        f_source.append(len(df_filter['user-screen_name'].drop_duplicates().tolist()) -
                        len(df_filter2['user-screen_name'].drop_duplicates().tolist()))

    # Plot Sankey diagram (outputs in a browser window)
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color='black',
                width=0.5),
            label=['Time-filtered dataset (' + str(sum(f_tweet[:2])) + '/' + str(sum(f_source[:2])) + ')', '',
                   'Source (' + str(sum(f_tweet[2:4])) + '/' + str(sum(f_source[2:4])) + ')', '', '',
                   'Risk-image (' + str(sum(f_tweet[4:6])) + '/' + str(sum(f_source[4:6])) + ')', '',
                   'Relevance (' + str(sum(f_tweet[6:8])) + '/' + str(sum(f_source[6:8])) + ')', '',
                   'Forecast (' + str(f_tweet[6]) + '/' + str(f_source[6]) + ')'],
            color=['black', 'black', 'black', '#b01e36', 'black', 'black', 'black', 'black', 'black', '#251fcf']),

        link=dict(
            source=[0, 0, 2, 2, 5, 5, 7, 7],
            target=[2, 3, 5, 3, 7, 3, 9, 3],
            value=f_tweet,
            color=['#b0afcc', '#dea9b1', '#b0afcc', '#dea9b1', '#b0afcc', '#dea9b1', '#b0afcc', '#dea9b1']))])

    fig.update_layout(title_text='Data Filtering (Tweet Count/Source Count)', font_size=40, font_color='black')
    fig.show()


# NEW Figure 3
def timeseries_ww_wwexp_nonww(df, images, freq, dates, median, show=True, save=False):
    # This function displays four subplots. The first subplot shows the number of tweets with watch/warning,
    # watch/warning (experimental), and non-watch/warning images over time as a stacked bar chart, where the width of
    # the bar corresponds to a user-provided time frequency (in minutes). The subsequent subplots display summary
    # diffusion information for each of the image groups. In each subplot, the total number of RTs and replies in each
    # time bin are plotted as black and red solid lines (respectively) and the median RT value is plotted as a circle,
    # where the color of the circle matches the color of the bars in the first subplot. User can choose whether to show
    # and/or save the figure (default is to show and not save).

    # Notes: Function assumes created-at column is not index of dataframe. Frequency must be provided in minutes as an
    # integer. Dates should be input as a list of timezone-aware datetime objects which correspond to the start-time of
    # the plot, the end-time of the plot, and any significant dates inbetween that should be highlighted. If user
    # chooses to save the plot, the figure will be saved to a "Timing" folder. If this folder does not exist, an error
    # will be raised.

    # Set the dataframe index to the created at column.
    df = df.set_index('tweet-created_at')

    # Format frequency as a string (for groupby) and convert to a string with the number of hours for the title.
    freq_str = str(freq) + 'Min'
    freq_title = str('{:0.0f}'.format(freq / 60)) + 'h'

    # Slice the tweet dataframe into three seperate dataframes which include only watch/warning (non-exp),
    # watch/warning (exp), and non-watch/warning images respectively.
    images_nonww = [image for image in images if (image != 'Watch/Warning') & (image != 'Watch/Warning (Exp)')]
    print(images_nonww)
    df_nonww = tweet_filter(df, image_range=images_nonww)
    df_ww = tweet_filter(df, image_range=['Watch/Warning'])
    df_ww_exp = tweet_filter(df, image_range=['Watch/Warning (Exp)'])

    # Calculate the tweet count for each sliced dataframe for each time bin, where the length of the bin is provided by
    # the user.
    count_nonww = df_nonww.groupby(pd.Grouper(freq=freq_str)).count()['diffusion-rt_count']
    count_ww = df_ww.groupby(pd.Grouper(freq=freq_str)).count()['diffusion-rt_count'].\
        reindex(count_nonww.index).fillna(0)
    count_ww_exp = df_ww_exp.groupby(pd.Grouper(freq=freq_str)).count()['diffusion-rt_count'].\
        reindex(count_nonww.index).fillna(0)

    # Calculate the total retweets for each sliced dataframe for each time bin.
    sum_rt_nonww = df_nonww.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-rt_count']
    sum_rt_ww = df_ww.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-rt_count']
    sum_rt_ww_exp = df_ww_exp.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-rt_count']

    # Calculate the total replies for each sliced dataframe for each time bin.
    sum_reply_nonww = df_nonww.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-reply_count']
    sum_reply_ww = df_ww.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-reply_count']
    sum_reply_ww_exp = df_ww_exp.groupby(pd.Grouper(freq=freq_str)).sum()['diffusion-reply_count']

    if median is True:
        # Calculate the median retweets for each sliced dataframe for each time bin.
        med_rt_nonww = df_nonww.groupby(pd.Grouper(freq=freq_str)).median()['diffusion-rt_count']
        med_rt_ww = df_ww.groupby(pd.Grouper(freq=freq_str)).median()['diffusion-rt_count']
        med_rt_ww_exp = df_ww_exp.groupby(pd.Grouper(freq=freq_str)).median()['diffusion-rt_count']

    # Create a figure and axes, then twin the axes so there are two y-axes.
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(11, 11))
    ax4 = ax1.twinx()
    ax5 = ax2.twinx()
    ax6 = ax3.twinx()
    axes = [ax0, ax1, ax2, ax3]

    if median is True:
        axes_twin = [ax4, ax5, ax6]

    # For each y-axis, share the bounds so that each subplot plots over the same y-values.
    ax1.get_shared_y_axes().join(ax1, ax2, ax3)

    if median is True:
        ax6.get_shared_y_axes().join(ax4, ax5, ax6)

    # Create a figure and set plotting variables.
    w = freq / 1800
    lw = 2
    a = 0.5
    fs = 14
    lp = 10
    ls = 12

    # Timing variables.
    start = min(dates)
    end = max(dates)
    td = timedelta(hours=(freq / 120 - 5))

    # Plot the non-watch/warning, watch/warning (non-exp), and watch/warning (exp) counts over time as a stacked bar
    # chart. Apply an offset to the x-axis (time) to ensure the bars line up properly.
    ax0.bar(count_nonww.index + td, count_nonww, width=w, color='blue', alpha=a, label='Non-W/W')
    ax0.bar(count_ww.index + td, count_ww, width=w, color='orange', alpha=a, label='W/W (Non-Exp)', bottom=count_nonww)
    ax0.bar(count_ww_exp.index + td, count_ww_exp, width=w, color='green', alpha=a, label='W/W (Exp)',
            bottom=np.array(count_nonww) + np.array(count_ww))

    # noinspection PyUnresolvedReferences
    pd.plotting.register_matplotlib_converters()

    # Line plots for total retweets.
    ax1.plot(sum_rt_ww_exp.index + td, sum_rt_ww_exp, color='black', linewidth=lw, label='Total RT')
    ax2.plot(sum_rt_ww.index + td, sum_rt_ww, color='black', linewidth=lw, label='Total RT')
    ax3.plot(sum_rt_nonww.index + td, sum_rt_nonww, color='black', linewidth=lw, label='Total RT')

    # Line plots for total replies.
    ax1.plot(sum_reply_ww_exp.index + td, sum_reply_ww_exp, color='maroon', linewidth=lw, label='Total Reply')
    ax2.plot(sum_reply_ww.index + td, sum_reply_ww, color='maroon', linewidth=lw, label='Total Reply')
    ax3.plot(sum_reply_nonww.index + td, sum_reply_nonww, color='maroon', linewidth=lw, label='Total Reply')

    if median is True:
        # Scatter plots for median retweet information.
        ax4.scatter(med_rt_ww_exp.index + td, med_rt_ww_exp, marker='o', facecolor='white', color='green',
                label='Median RT')
        ax5.scatter(med_rt_ww.index + td, med_rt_ww, marker='o', facecolor='white', color='orange', label='Median RT')
        ax6.scatter(med_rt_nonww.index + td, med_rt_nonww, marker='o', facecolor='white', color='blue', label='Median RT')

    # Format major axes and labels.
    for ax in axes:
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=ls)
        ax.tick_params(axis='y', labelsize=ls)
        ax.set_ylabel('Total Diffusion', fontsize=fs, labelpad=lp)
        ax.legend(loc='upper left', fontsize=ls)
    ax0.set_ylabel('Tweet Count', fontsize=fs, labelpad=lp)

    if median is True:
        # Format secondary axes and labels.
        for ax in axes_twin:
            ax.tick_params(axis='y', labelsize=ls)
            ax.set_ylabel('Median RT', fontsize=fs, labelpad=lp)
            ax.legend(loc='upper right', fontsize=ls)

    # Add light gray lines at times of significance.
    date_plot = [x for x in dates if x != max(dates) and x != min(dates)]
    for ax in axes:
        for date in date_plot:
            ax.axvline(date, color='gray', alpha=0.5)

    # Format figure for plotting.
    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig.savefig('timeseries_6h.png', dpi=300)


# Figure 4
def rt_rate_plot(df_all, gb, metric, show=True, save=False, **kwargs):
    # This function plots the cumulative median diffusion at a set of pre-calculated times after tweet creation (e.g.
    # 5 mins, 30 mins, 2h, 6h). The function plots one timeseries for each unique element in the user-provided gb
    # column name (e.g. 'image-type' or 'user-scope_aff'). If the user chooses 'image-type' as the grouby column, they
    # can optionally provide a 'df_nokey' tweet dataframe (a tweet dataframe with key messages graphics filtered out) as
    # a keyword argument. If provided, the function will create two subplots - one with all image types, and one with
    # key message removed. The metric input controls which diffusion metric is plotted (e.g. 'rt', 'reply', 'qt'). The
    # user can choose whether to show and/or save the figure (default is to show and not save).

    # Notes: Function assumes that input dataframe does not already have 'tweet-created_at' as the index. Frequency must
    # be provided as an integer, in minutes (e.g. for 3h, input 180). Metric must match the form used in the column
    # name (retweet: 'rt', reply: 'reply', quote tweet: 'qt'). If user chooses to save output, note that output
    # is saved to a "Rates" folder. If this folder does not exist, an error will be raised.

    # Gather all the diffusion count columns for the user-provided metric.
    count_cols = []
    for col in df_all.columns:
        if ((('diffusion-' + metric) in col) is True) & (('count' in col) is True):
            count_cols.append(col)

    # Obtain 'nokey' dataframe, if provided.
    df_nokey = kwargs.get('df_nokey', None)

    # Calculate the median value for each retweet diffusion column for each image or source grouping for the all image
    # dataframe.
    df1_gb = df_all.groupby([gb])[count_cols].median()

    # Sort the groupby by the 6h count column (the latest time value, other than the final one), transpose so that
    # each column represents one source/image group, remove the final row, and reformat index so that it displays
    # the number of minutes the diffusion value is calculated over (to make plotting more intuitive).
    df1_gb.sort_values('diffusion-' + metric + '_count_360m', ascending=False, inplace=True)
    df1_gb = df1_gb.T
    df1_gb = df1_gb.drop('diffusion-' + metric + '_count', axis=0)
    df1_gb.index = [5, 10, 15, 30, 60, 120, 240, 360]

    if gb == 'image-type':
        # Obtain RGBA color values from the Tab 10 color scheme (the Matplotlib default).
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Convert the RGBA values to RGB, then to hex. Append the first and second hex colors twice to account for the
        # twelve image categories.
        color_hex = []
        for color in colors:
            rgb = color[:3]
            color_hex.append(rgb2hex(rgb))
        color_hex.append(color_hex[0])
        color_hex.append(color_hex[1])

        # If user includes a 'nokey' dataframe (the same dataframe but with key messages graphics removed), plot a
        # second subplot for the no key messages data. Make the image twice as wide to compensate.
        if df_nokey is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            # Calculate the median value for each retweet diffusion column for each image for the nokey dataframe.
            df2_gb = df_nokey.groupby([gb])[count_cols].median()

            # Sort the groupby by the 6h count column (the latest time value, other than the final one), transpose so
            # that each column represents one image group, remove the final row, and reformat index so that it displays
            # the number of minutes the diffusion value is calculated over (to make plotting more intuitive).
            df2_gb.sort_values('diffusion-' + metric + '_count_360m', ascending=False, inplace=True)
            df2_gb = df2_gb.T
            df2_gb = df2_gb.drop('diffusion-' + metric + '_count', axis=0)
            df2_gb.index = [5, 10, 15, 30, 60, 120, 240, 360]

            # Plot the groupby for all images, using all the color values (the original Tab 10, plus the looped 2).
            # Include title, xlabel, ylabel, and legend.
            df1_gb.plot(ax=axes[0], color=color_hex)
            axes[0].set_title('Median retweet diffusion over time by image')
            axes[0].set_xlabel('Minutes since post', labelpad=10)
            axes[0].set_ylabel('Cumulative retweets', labelpad=20)

            # Plot the groupby for the nokey dataframe. Conserve the color scheme from the first plot by only using the
            # colors for the n plot items the two plots share. Include title and xlabel (but no ylabel).
            df2_gb.plot(ax=axes[1], color=color_hex[(len(df1_gb.columns) - len(df2_gb.columns)):])
            axes[1].set_title('Median retweet diffusion over time by image (no key messages)')
            axes[1].set_xlabel('Minutes since post', labelpad=10)

            # Create one legend for both subplots, anchored below the subplots and roughly centered. Adjust subplots to
            # make room for legend.
            axes[0].legend(loc='upper left', bbox_to_anchor=[0.5, -0.125], ncol=3, fontsize=8)
            axes[1].get_legend().remove()
            fig.subplots_adjust(bottom=0.2)  # , left=0.25, right=0.9)

        # If no nokey df is provided, only plot with one subplot.
        else:
            fig, ax = plt.subplots(figsize=(7, 7))

            # Plot the groupby for all images, using all the color values (the original Tab 10, plus the looped 2).
            # Include title, xlabel, ylabel, and legend.
            df1_gb.plot(ax=ax, color=color_hex)
            ax.set_title('Median retweet diffusion over time by image')
            ax.set_xlabel('Minutes since post', labelpad=10)
            ax.set_ylabel('Cumulative retweets', labelpad=20)
            ax.legend(loc='upper left', ncol=2, fontsize=8)

        # Show figure, if desired.
        if show is True:
            plt.show()

        # Save figure, if desired.
        if save is True:
            fig.savefig('Rates\\rt_rate_image.png', dpi=300)

    # If creating a source grouped plot, create only one subplot for all sources (using the all images dataframe).
    # Include title, xlabel, ylabel, and legend below the plot, adjusting subplot to make room. Save figure.
    elif gb == 'user-scope_aff':
        fig, ax = plt.subplots(figsize=(7, 7))
        df1_gb.plot(ax=ax)
        ax.set_title('Median retweet diffusion over time by source')
        ax.set_xlabel('Minutes since post', labelpad=10)
        ax.set_ylabel('Cumulative retweets', labelpad=20)
        ax.legend(loc='upper left', bbox_to_anchor=[-0.05, -0.125], ncol=4, fontsize=8)
        fig.subplots_adjust(bottom=0.2)  # , left=0.2, right=0.9)

        # Show figure, if desired.
        if show is True:
            plt.show()

        # Save figure, if desired.
        if save is True:
            fig.savefig('Rates\\rt_rate_source.png', dpi=300)


# Figure 5/7 - not created yet


# Figure 6/8
def timeline(df, value_cols, values, labels, size_col, color_col, dates, zeros=True, show=True, save=False):
    # This function creates a figure from a user-provided tweet dataframe that displays each tweet as a circle, where
    # the size of the circle represents the value of the user-provided size column (e.g. diffusion-rt_count) and the
    # color represents the value of a binary user-provided color column (e.g. image-branding_off). Circles are plotted
    # in rows, where each row represents one value of a user-provided value column (e.g. image-type). User provides
    # the values they'd like to include, in the order they'd like to include them, using the values input. User can
    # choose to show zero values on the plot or to suppress them (default is to show). User can choose whether to show
    # and/or save the figure (default is to show and not save).

    # Notes: the color-col MUST be a binary column. Note that the function plots from the bottom up. Therefore, the
    # values should be input in reverse order of how the user would like them to read top-down. Function assumes
    # created-at column is not index of dataframe. Dates should be input as a list of timezone-aware datetime objects
    # which correspond to the start-time of the plot and the end-time of the plot. Other dates can be included, but they
    # will not be highlighted, as in other plots. If user chooses to save the plot, the figure will be saved to a
    # "Timing" folder. If this folder does not exist, an error will be raised.

    # Create a figure and set plotting variables.
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ec = 'black'
    lw = 0.5
    a = 0.5
    fs = 14
    ls = 12
    y = 1
    ydelta = 0.1
    yticks = []

    # Timing variables.
    start = min(dates)
    end = max(dates)

    # Set color, labels, and titles based off user-provided color column. If the user-provided column does not match
    # the existing options, allow them to input which values they'd like.
    if color_col == 'image-branding_off':
        c1 = '#603F83'
        d1 = 'Official'
        c2 = '#E9738D'
        d2 = 'Unofficial'
        title = 'branding'

    elif color_col == 'image-type_multi':
        c1 = 'blue'
        d1 = 'Overlap'
        c2 = 'white'
        d2 = 'No Overlap'
        title = 'multi'

    elif color_col == 'user-scope_loc':
        c1 = '#2CAE66'
        d1 = 'Local'
        c2 = '#FFA177'
        d2 = 'National'
        title = 'scope'

    elif color_col == 'user-agency_ind':
        c1 = '#FC766A'
        d1 = 'Individual'
        c2 = '#5B84B1'
        d2 = 'Organization'
        title = 'agency'

    else:
        print('Your selected color column: ' + color_col)
        c1 = input('Please provide a color to represent the first value.')
        d1 = input('Please provide a label to represent the first value.')
        c2 = input('Please provide a color to represent the second value.')
        d2 = input('Please provide a label to represent the second value.')
        title = input('Please provide a shorthand for the color column to be used in the image title.')

    # Set size title based on size column. If the user-provided size column does not match the options given, allow them
    # to choose the title based on their chosen column.
    if size_col == 'diffusion-rt_count':
        size_title = 'rt'
    elif size_col == 'diffusion-reply_count':
        size_title = 'reply'
    else:
        print('Your selected size column: ' + size_col)
        size_title = input('Please provide a shorthand for the size column to be used in the image title.')

    # Set value to add to size column, depending on whether user chooses to visualize zeros or not.
    if zeros is True:
        add = 1
    else:
        add = 0
    
    # Loop over each unique value in the user-provided value column.
    for col in value_cols:
        for value in values:
            # Split the dataframe in to two based on the value of the color column.
            tweets_c1 = df.loc[(df[col] == value) & (df[color_col] == 1)]
            tweets_c2 = df.loc[(df[col] == value) & (df[color_col] != 1)]

            # Add a value to the size column depending on whether the user chooses to visualize zeros or not (the add value
            # is 1 if zeros are included and 0 if not).
            tweets_c1[size_col] = tweets_c1[size_col] + add
            tweets_c2[size_col] = tweets_c2[size_col] + add

            # Obtain the sorted dates for each split dataframe.
            tweets_c1.sort_values(by='tweet-created_at', inplace=True)
            times_c1 = tweets_c1['tweet-created_at']
            tweets_c2.sort_values(by='tweet-created_at', inplace=True)
            times_c2 = tweets_c2['tweet-created_at']

            # Plot the split dataframes at the same y-value (all in one line), where the x-value is the time the tweet was
            # posted. Size the circles by the user-provided size column. Vary the split dataframes by color.
            ax.scatter(times_c1, [y] * len(times_c1), alpha=a,
                       s=tweets_c1[size_col], edgecolor=ec, linewidth=lw, c=c1)
            ax.scatter(times_c2, [y] * len(times_c2), alpha=a,
                       s=tweets_c2[size_col], edgecolor=ec, linewidth=lw, c=c2)

            # Append the y-value to the yticks array.
            yticks.append(y)

            # Increment y by ydelta before moving on to the next value.
            y += ydelta

    # Set xticks and xticklabels by hand so they correspond to local time (UTC -5). Draw a vertical line at midnight
    # local time for each day in the time range.
    time_delta = timedelta(days=1)
    xticks = []
    xticklabels = []
    ax.set_xlim(start, end)
    while start <= end:
        xticks.append(start)
        xticklabels.append(start.strftime('%b-%d'))
        plt.axvline(start, c='gray', alpha=0.25)
        start += time_delta
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, ha='right')
    ax.tick_params(axis='x', labelrotation=45, labelsize=ls)

    # Set yticks and yticklabels to represent each value. Remove yticks so only labels remain.
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', labelsize=fs, length=0)

    # Remove borders on right, left, and top of image.
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Manually set legend for color values.
    legend_elements1 = [Patch(facecolor=c1, edgecolor='black', label=d1),
                        Patch(facecolor=c2, edgecolor='black', label=d2)]
    fig.legend(handles=legend_elements1, loc='lower center', bbox_to_anchor=[0.75, 0.05], ncol=4, fontsize=ls)

    # Manually set legend for size values.
    size = [10, 100, 1000]
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[0]),
               markersize=math.sqrt(size[0])),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[1]),
               markersize=math.sqrt(size[1])),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[2]),
               markersize=math.sqrt(size[2]))]
    ax.legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=[0, -0.125], ncol=4, borderpad=1.5,
              fontsize=ls)
    fig.subplots_adjust(bottom=0.2, left=0.27, right=0.95)

    # Save figure, if desired.
    if save is True:
        fig.savefig('Timing\\timeline_' + title + '_' + size_title + '.png', dpi=300)

    # Show figure, if desired.
    if show is True:
        plt.show()


# Figure 9
def timeseries_nonww(df, freq, images, dates, show=True, save=False):
    # This function displays the tweet count and diffusion patterns for a set of user-provided image types over time.
    # The first subplot shows the count of each image type over time as a stacked bar chart, where the width of the bar
    # corresponds to a user-provided time frequency (in minutes). The function will plot each of the user-provided image
    # types, in addition to an "Other" category, which groups all remaining/non-selected image types.

    # The subsequent subplots display summary diffusion information for each image type (including the 'Other' group).
    # In each subplot, the total number of RTs and replies in each time bin are plotted as black and red solid lines
    # (respectively) and the median RT value is plotted as a circle, where the color of the circle matches the color of
    # the bars in the first subplot. User can choose whether to show and/or save the figure (default is to show and not
    # save).

    # Notes: the function will replace all image types not specified in the user-input as "Other". This includes
    # Watch/Warning images if inputting the full, final dataset. To work around this, first filter the dataset to remove
    # watch/warnings, and then run the function. Function assumes created-at column is not index of dataframe. Frequency
    # must be provided in minutes as an integer. Dates should be input as a list of timezone-aware datetime objects
    # which correspond to the start-time of the plot, the end-time of the plot, and any significant dates inbetween
    # that should be highlighted. If user chooses to save the plot, the figure will be saved to a "Timing" folder. If
    # this folder does not exist, an error will be raised.

    # Set the dataframe index to the created at column.
    df = df.set_index('tweet-created_at')

    # Format frequency as a string (for groupby) and convert to a string with the number of hours for the title.
    freq_str = str(freq) + 'Min'
    freq_title = str('{:0.0f}'.format(freq / 60)) + 'h'

    # Replace all image types not included in the user-input list as "Other".
    all_images = df['image-type'].drop_duplicates().tolist()
    images_replace = [image if image in images else 'Other' for image in all_images]
    df['image-type'].replace(dict(zip(all_images, images_replace)), inplace=True)
    new_images = ['Other'] + images

    # Count the number of tweets with each user-provided image type in each time bin of user-provided frequency. Replace
    # NaN values with zero.
    count = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).count()['tweet-id_trunc'].unstack()
    count = count[new_images]
    count = count.fillna(0)

    # Count the total number of retweets for each image type for each time bin.
    rt_sum = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).sum()['diffusion-rt_count'].unstack()
    rt_sum = rt_sum.fillna(0)

    # Count the total number of replies for each image type for each time bin.
    reply_sum = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).sum()['diffusion-reply_count'].unstack()
    reply_sum = reply_sum.fillna(0)

    # Calculate the median retweet value for each image type for each time bin.
    rt_med = df.groupby([pd.Grouper(freq=freq_str), 'image-type']).median()['diffusion-rt_count'].unstack()

    # Create a figure and set plotting variables.
    fig = plt.figure(figsize=(11, 11))
    pd.plotting.register_matplotlib_converters()
    colors = ['gray', 'blue', 'hotpink', 'green', 'red', 'purple', 'red', 'gray']
    w = freq / 1800
    lw = 2
    a = 0.5
    lp = 10

    # Set font and label sizes based on the number of images input by user.
    if len(images) >= 6:
        fs = 10
        ls = 8

    elif len(images) >= 4:
        fs = 12
        ls = 10

    else:
        fs = 14
        ls = 12

    # Timing variables.
    start = min(dates)
    end = max(dates)
    td = timedelta(hours=(freq / 120 - 5))

    # Plot the tweet count for each image type for each time bin as a stacked bar chart. Apply an offset to the x-axis
    # (time) to ensure the bars line up properly.
    ax0 = fig.add_subplot(len(new_images)+1, 1, 1)
    bottom = np.zeros_like(np.array(count['Cone']))
    for i, col in enumerate(count.columns):
        ax0.bar(count.index + td, count[col], width=w, color=colors[i], alpha=a, label=col, bottom=bottom)
        bottom += np.array(count[col])

    # Format y-axis and label.
    ax0.set_ylabel('Tweet Count', fontsize=fs, labelpad=lp)
    ax0.set_ylim(0, 60)

    # Store axis for formatting later.
    axes = [ax0]

    # Plot the total retweet and reply diffusion as red and maroon lines, respectively. The first diffusion subplot must
    # be created outside the loop below in order to create a y-axis that can be shared among all subplots.
    ax1 = fig.add_subplot(len(new_images) + 1, 1, 2)
    ax1.plot(rt_sum.index + td, rt_sum[new_images[0]], color='black', linewidth=lw, label='Total RT')
    ax1.plot(reply_sum.index + td, reply_sum[new_images[0]], color='maroon', linewidth=lw, label='Total Reply')

    # On the same subplot, but using a secondary y-axis, plot the median retweet diffusion as a circle with a white face
    # and a colored outline.
    ax_twin1 = ax1.twinx()
    ax_twin1.scatter(rt_med.index + td, rt_med[new_images[0]], marker='o', facecolor='white', color=colors[0],
                     label=new_images[0])

    # Format y-axis label.
    ax1.set_ylabel('Total Diffusion', fontsize=fs, labelpad=lp)

    # Obtain the maximum y-value of the secondary y-axis and store for formatting later.
    yt_min, yt_max = ax_twin1.get_ylim()
    maxes = [yt_max]

    # Store the primary and secondary axes for formatting later.
    axes.append(ax1)
    axes_twin = [ax_twin1]

    # For all other user-provided image types...
    for i, image in enumerate(new_images[1:]):
        # Plot the total retweet and reply diffusion. Use shared y-axis to plot each subplot over the same primary
        # y-range.
        ax = fig.add_subplot(len(new_images)+1, 1, i+3, sharey=ax1)
        ax.plot(rt_sum.index + td, rt_sum[image], color='black', linewidth=lw, label='Total RT')
        ax.plot(reply_sum.index + td, reply_sum[image], color='maroon', linewidth=lw, label='Total Reply')

        # Plot the median retweet diffusion.
        ax_twin = ax.twinx()
        ax_twin.scatter(rt_med.index + td, rt_med[image], marker='o', facecolor='white', color=colors[i + 1],
                        label=image)

        # Format the y-axis label and legend.
        ax.set_ylabel('Total Diffusion', fontsize=fs, labelpad=lp)
        ax.legend(loc='upper left', fontsize=ls)

        # Store the maximum value of the secondary axis for formatting later.
        yt_min, yt_max = ax_twin.get_ylim()
        maxes.append(yt_max)

        # Store the primary and secondary axes for formatting later.
        axes.append(ax)
        axes_twin.append(ax_twin)

    # Format primary axes and labels.
    for ax in axes:
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=ls)
        ax.tick_params(axis='y', labelsize=ls)

    # Format secondary axes and labels.
    for ax in axes_twin:
        # Plot the secondary axes over the same range by setting the maximum value for each subplot equal to the maximum
        # value of all subplots.
        ax.set_ylim(0, max(maxes))
        ax.tick_params(axis='y', labelsize=ls)
        ax.set_ylabel('Median RT', fontsize=fs, labelpad=lp)
        ax.legend(loc='upper right', fontsize=ls)

    # Add light gray lines at times of significance.
    date_plot = [x for x in dates if x != max(dates) and x != min(dates)]
    for ax in axes:
        for date in date_plot:
            ax.axvline(date, color='gray', alpha=0.5)

    # Format figure for plotting.
    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig.savefig('Timing\\timeseries_nonww_' + freq_title + '.png', dpi=300)


# Other figures
def timeseries_all(df, metrics, stat, dates, date_labels, freq, suppress_val=5, outlier=True, show=True, save=False):
    # This function produces a plot of various metrics plotted over time. The function takes a tweet dataframe (which
    # includes diffusion metrics summarized for each tweet), a number of metrics to summarize (such as RT, reply,
    # or QT), the statistic used to summarize the data for each bin (mean or median), dates (a series of
    # timezone-aware datetime objects which correspond to the start-time of the plot, the end-time of the plot, and any
    # significant dates inbetween that should be highlighted), date_labels (the text annotation labels for the
    # significant dates included), and finally, the frequency over which statistics should be binned (e.g. 3 hours,
    # 6 hours).

    # The user can choose whether to suppress output from plotting if the tweet count in a time bin is less
    # than a user-provided suppression value (default is five). The user can also note whether the outlier tweet is
    # present in the input dataframe (default is true). If so, the outlier time bin will be annotated if the user
    # chooses to summarize the mean diffusion values over time. The user can choose to show the plot and/or save the
    # figure (default is to show the figure and not save).

    # Notes: Metrics must be input exactly as they appear in the column names (e.g. 'rt' = retweet, 'reply' = reply,
    # 'qt' = quote tweet). Frequency must be provided in minutes as an integer. If user chooses to save the plot, the
    # figure will be saved to a "Timing" folder. If this folder does not exist, an error will be raised.

    # Format frequency as a string with the number of hours for the title.
    freq_title = str('{:0.0f}'.format(freq / 60)) + 'h'

    # Timing variables.
    start = min(dates)
    end = max(dates)
    td = timedelta(minutes=freq)

    # Remove the largest and smallest values in the dates list, yielding the remaining dates that should be represented
    # visually in the plot.
    date_plot = [x for x in dates if x != max(dates) and x != min(dates)]

    # Create a figure object with a variable number of subplots based on the number of metrics given (+1 to include a
    # tweet count subplot). Set edgecolor and color values, bar width and linediwth variables, along with text/axis
    # variables.
    plot_num = len(metrics)
    fig, axes = plt.subplots(plot_num + 1, 1, figsize=(11, 11))
    ec = 'black'
    colors = ('#fbb4ae', '#b3cde3', '#ccebc5')
    fs = 16
    rot = 0
    lp = 40
    w = freq / 1800
    lw = freq / 360

    # Create a subplot count and establish empty arrays to fill in loops.
    n = 0
    vals = []
    times = []
    plot_times = []

    # Calculate the number of tweets within each n-hour bin from the start to the end time. Append time values for each
    # iteration through the loop in order to calculate binned values later. Additionally, create "plot times" which
    # add delta/2 hours so that they display at the midpoint of the time bin (e.g. 0 to 6 would be plotted at 3), then
    # subtract 5 to account for timezone difference between data in Central time and mdates axis plotting method in UTC.
    start_tz_plot = start
    while start_tz_plot <= end:
        vals.append(len(df.loc[(df['tweet-created_at'] >= start_tz_plot) &
                               (df['tweet-created_at'] < start_tz_plot + td)]))
        times.append(start_tz_plot)
        plot_times.append(start_tz_plot + timedelta(hours=(freq/120 - 5)))
        start_tz_plot += td

    # Plot tweet count values in first subplot. Increase subplot count by one.
    axes[n].bar(plot_times, vals, color='white', edgecolor=ec, linewidth=lw, width=w)
    axes[n].set_ylabel('Tweet\nCount', fontsize=fs, rotation=rot, labelpad=lp)

    # Create labels for each of the timestamps provided (other than start and end).
    for i in range(0, len(date_plot)):
        axes[n].annotate(date_labels[i], xy=(date_plot[i] - timedelta(hours=2), 0.82 * max(vals)), ha='right')

    # Calculate the mean/median metric value for each time-bin. If the time-bin has fewer tweets than user-provided
    # suppression value, suppress the output.
    n = 1
    for metric in metrics:
        vals = []
        for time in times:
            if len(df.loc[(df['tweet-created_at'] >= time) & (df['tweet-created_at'] < time + td)]) <= suppress_val:
                vals.append(0)
            else:
                if stat == 'mean':
                    vals.append(df.loc[(df['tweet-created_at'] >= time) & (df['tweet-created_at'] < time + td)]
                                ['diffusion-'+str(metric)+'_count'].mean())
                elif stat == 'median':
                    vals.append(df.loc[(df['tweet-created_at'] >= time) & (df['tweet-created_at'] < time + td)]
                                ['diffusion-' + str(metric) + '_count'].median())

        # Plot the summarized values for each time bin.
        axes[n].bar(plot_times, vals, color=colors[n - 1], edgecolor=ec, linewidth=lw, width=w)

        # Label the y-axes.
        axes[n].set_ylabel(stat.capitalize() + '\n' + metric.capitalize(), fontsize=fs, rotation=rot, labelpad=lp)

        # If user summarizes using mean and notes that outlier is included...
        if (stat == 'mean') & (outlier is True):
            # Find the outlier value (corresponding to the Brazoria County tweet). Round the outlier value to the
            # nearest integer for plotting purposes.
            outlier_id = int(np.argmax(vals))
            outlier_val = round(vals[outlier_id])

            # Remove the outlier time period and find the maximum of the remaining time bins. Round this value to the
            # nearest integer for plotting purposes and to the nearest 5 (for replies) or 10 (for retweets) for setting
            # the y-axis.
            del vals[outlier_id]
            second_highest = int(np.argmax(vals))
            second_highest_val = round(vals[second_highest])
            if metric == 'rt':
                base = 10
            else:
                base = 5

            def myround(x):
                return base * math.ceil(x / base)
            second_highest_val_rounded = myround(second_highest_val)

            axes[n].set_ylim(0, second_highest_val_rounded)

            # Create an annotation to the outlier data bar, labeled with the outlier value, placed to the right of the
            # data bar, with an arrow pointing up towards the y-limit.
            axes[n].annotate(str(outlier_val), xy=(plot_times[outlier_id], 0.97*second_highest_val_rounded),
                             xytext=(plot_times[outlier_id + int(720/freq)], 0.8*second_highest_val_rounded),
                             arrowprops=dict(facecolor='black', width=0.5, headwidth=8))

        n += 1

    # Set the x-axis to show each day (technically in GMT but we countered this earlier by adjusting the plot times).
    for i in range(0, plot_num + 1):
        axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=1))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        axes[i].tick_params(axis='x', labelrotation=45)
        for label in axes[i].get_xticklabels():
            label.set_horizontalalignment('right')
        axes[i].tick_params(labelsize=14)
        axes[i].set_xlim(start - timedelta(hours=6), end - timedelta(hours=3))

        # Add light gray lines at times of significance.
        for date in date_plot:
            axes[i].axvline(date, c='gray')

    # Final figure formatting.
    plt.tight_layout()

    # Show the figure, if desired.
    if show is True:
        plt.show()

    # Save the figure, if desired.
    if save is True:
        fig.savefig('Timing\\timeseries_all_' + stat + '_' + freq_title + '_bar.png', dpi=300)


def diffusion_curves(tweet_df, diff_folder, show=True, save=False):
    # This function creates five plots that may be of use in the initial stages of assessing the patterns of retweet
    # diffusion over time. The first plot is of "diffusion curves" - e.g. for each tweet, the amount of retweets at each
    # time after the original tweet was posted (up to 4 hours after creation). The second plot is a histogram of the
    # count of retweets in each 3-minute time bin after the tweet was posted (up to 240 minutes). The third plot is the
    # same as the second plot, except the histogram is cumulative. The final two plots display the mean number of
    # retweets accrued at each minute after the tweet was posted (up to 240 minutes) - the first displays the mean
    # count, the second displays the percentage of the final retweet value accrued at each minute.

    # Each plot divides the data into quintiles of the final retweet distribution to assess how changes in diffusion
    # over time vary by how many retweets a tweet receives over its entire lifetime. This is useful in assessing
    # whether certain rate or count metrics bias towards certain groups, or whether these statistics offer any
    # explanatory power beyond what the final retweet count offers.

    # This function requires two inputs: a tweet dataframe, which includes data for each tweet, and a diffusion files
    # folder, where JSON files are stored for each tweet, containing the diffusion (retweets/quote tweets/replies) of
    # the original tweet. The index of the tweet dataframe CANNOT BE the tweet-created-at column. The user can choose
    # whether they'd like to show or save the figures (the defaults are set to show but not save). If the user chooses
    # to save the output, the function saves it to a "Rates" folder. If a "Rates" folder does not exist, an error will
    # be raised.

    # IMPORTANT: Because this function requires data for each minute for each tweet, it takes several minutes to run.

    # Create empty dataframes for storing retweet data and a minutes counter to loop over.
    minutes = np.arange(0, 240)
    all_rt = pd.DataFrame()
    all_rt_min = pd.DataFrame()

    # Create the plot for diffusion curves and define plotting variables.
    fig1, axes1 = plt.subplots(3, 2, figsize=(10, 9))
    ax_fs = 12
    label_fs = 14
    title_fs = 16

    # Calculate quintiles of the final retweet distribution.
    rt_final_quint = tweet_df['diffusion-rt_count'].quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()

    # Read diffusion data for each tweet-id in diffusion file folder (where the diffusion of each tweet is stored as a
    # seperate JSON file).
    for filename in os.listdir(diff_folder):
        with open(diff_folder + '\\' + filename, 'r') as f:
            json_data = json.load(f)

        # Convert data for each tweet-id in to a DataFrame (but only if tweet has any diffusion).
        if len(json_data) != 0:
            diff_data_df = pd.json_normalize(json_data)

            # Rename created-at column.
            diff_data_df.rename(columns={'created_at.$date': 'timestamp'}, inplace=True)

            # Only include retweet diffusion (and not replies).
            rt_data_df = diff_data_df.loc[diff_data_df['tweet_types.retweet'] != 0]

            # Exclude outlier tweet.
            if (len(rt_data_df) > 0) & (len(rt_data_df) <= 10000):

                # Convert timestamp to a datetime object and set as index.
                rt_data_df['timestamp'] = pd.to_datetime(rt_data_df['timestamp'])
                rt_data_df = rt_data_df.set_index('timestamp')

                # Obtain the tweet created-at time from tweet-data, matching tweet-data id to the current filename
                # iteration (excluding the .json). TWEET CREATED AT MUST NOT BE SET AS INDEX.
                created_at = tweet_df.loc[tweet_df['tweet-id_trunc'].astype(str) == filename[:15],
                                          'tweet-created_at'].iloc[0].tz_convert('UTC')
                print(created_at)

                # For each RT for the current tweet-id, calculate the time delta between the tweet's creation and the
                # time the RT occurred. Convert to minutes.
                rt_data_df['delta'] = rt_data_df.index - created_at
                rt_data_df['delta'] = (rt_data_df['delta'].dt.days * 1440) + (rt_data_df['delta'].dt.seconds / 60)

                # Sort the dataframe by delta to order the RTs.
                rt_data_df.sort_values('delta', inplace=True)

                # Append a RT value column to the dataframe, based on sorted order. Add 1 since retweet 1 is in
                # column 0.
                rt_num = np.arange(0, len(rt_data_df))
                rt_data_df['rt_num'] = rt_num + 1

                # Append a final RT value column to the dataframe.
                rt_val = [len(rt_data_df)] * len(rt_data_df)
                rt_data_df['rt_final'] = rt_val

                # Plot diffusion curves (one if loop for each quintile range to be plotted). Plot occurs in loop so that
                # curves are created for each tweet.
                if 0 < len(rt_data_df) <= rt_final_quint[0]:
                    axes1[0, 0].plot(rt_data_df['delta'], rt_data_df['rt_num'], color='blue', alpha=0.5)
                elif rt_final_quint[0] < len(rt_data_df) <= rt_final_quint[1]:
                    axes1[0, 1].plot(rt_data_df['delta'], rt_data_df['rt_num'], color='orange', alpha=0.5)
                elif rt_final_quint[1] < len(rt_data_df) <= rt_final_quint[2]:
                    axes1[1, 0].plot(rt_data_df['delta'], rt_data_df['rt_num'], color='green', alpha=0.5)
                elif rt_final_quint[2] < len(rt_data_df) <= rt_final_quint[3]:
                    axes1[1, 1].plot(rt_data_df['delta'], rt_data_df['rt_num'], color='red', alpha=0.5)
                elif len(rt_data_df) > rt_final_quint[3]:
                    axes1[2, 0].plot(rt_data_df['delta'], rt_data_df['rt_num'], color='purple', alpha=0.5)

                # Create a dataframe that includes all retweets for for all tweets for macro analysis.
                all_rt = all_rt.append(rt_data_df)

                # Loop over each of the first 240 minutes of retweeting to count how many RTs occur in each minute,
                # calculated for each tweet.
                rt_min = []
                for minute in minutes:
                    rt_min.append(len(rt_data_df.loc[(rt_data_df['delta'] >= minute) & (rt_data_df['delta'] <
                                                                                        minute + 1)]))

                # Create a dataframe of minute data, including the number of RTs/minute, the final RTs (static for each
                # row), and a percent of total column calculated from the previous columns.
                rt_val = [len(rt_data_df)] * len(minutes)
                rt_min_df = pd.DataFrame({'rt_final': rt_val, 'minute': minutes, 'rt_count': rt_min})
                rt_min_df['rt_per'] = (rt_min_df['rt_count'] / rt_min_df['rt_final']) * 100

                # Create a dataframe with retweet count and percentage of total values for each tweet for each
                # minute (up to 240) after the tweet was posted.
                all_rt_min = all_rt_min.append(rt_min_df)

    # Title diffusion curve subplots with quintile ranges. Set final subplot to not plot.
    axes1[0, 0].set_title('Final RT: [1,' + str(rt_final_quint[0]) + ']', fontsize=title_fs)
    axes1[0, 1].set_title('Final RT: (' + str(rt_final_quint[0]) + ',' + str(rt_final_quint[1]) + ']',
                          fontsize=title_fs)
    axes1[1, 0].set_title('Final RT: (' + str(rt_final_quint[1]) + ',' + str(rt_final_quint[2]) + ']',
                          fontsize=title_fs)
    axes1[1, 1].set_title('Final RT: (' + str(rt_final_quint[2]) + ',' + str(rt_final_quint[3]) + ']',
                          fontsize=title_fs)
    axes1[2, 0].set_title('Final RT: (' + str(rt_final_quint[3]) + ',' + str(rt_final_quint[4]) + ']',
                          fontsize=title_fs)
    axes1[2, 1].set_visible(False)

    # Set tick and axis labels.
    for x in range(0, 3):
        for y in range(0, 2):
            axes1[x, y].xaxis.set_tick_params(labelsize=ax_fs)
            axes1[x, y].yaxis.set_tick_params(labelsize=ax_fs)
            axes1[x, y].set_xlabel('Minutes Since Post', fontsize=label_fs)
            axes1[x, y].set_ylabel('Cumulative RTs', fontsize=label_fs)

    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig1.savefig('Rates\\rt_curves_quint.png', dpi=300)

    # Non-cumulative histogram plot for retweet count at each minute, plotted seperately for each quintile.
    fig2, axes2 = plt.subplots(3, 2, figsize=(10, 9))

    # Select all data between the qunitile bounds and for which the time between tweet creation and retweet is less than
    # 240 minutes. Plot as non-cumulative histogram with three-minute bins.
    all_rt['delta'].loc[
        (all_rt['rt_final'] > 0) & (all_rt['rt_final'] <= rt_final_quint[0]) & (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=False, grid=False, color='blue', edgecolor='black', ax=axes2[0, 0])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[0]) & (all_rt['rt_final'] <= rt_final_quint[1]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=False, grid=False, color='orange', edgecolor='black', ax=axes2[0, 1])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[1]) & (all_rt['rt_final'] <= rt_final_quint[2]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=False, grid=False, color='green', edgecolor='black', ax=axes2[1, 0])

    all_rt['delta'].loc[
        (all_rt['rt_final'] > rt_final_quint[2]) & (all_rt['rt_final'] <= rt_final_quint[3]) &
        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=False, grid=False, color='red', edgecolor='black', ax=axes2[1, 1])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[3]) & (all_rt['rt_final'] <= rt_final_quint[4]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=False, grid=False, color='purple', edgecolor='black', ax=axes2[2, 0])

    # Title non-cumulative histogram subplots with quintile ranges. Set final subplot to not plot.
    axes2[0, 0].set_title('Final RT: [1,' + str(rt_final_quint[0]) + ']', fontsize=title_fs)
    axes2[0, 1].set_title('Final RT: (' + str(rt_final_quint[0]) + ',' + str(rt_final_quint[1]) + ']',
                          fontsize=title_fs)
    axes2[1, 0].set_title('Final RT: (' + str(rt_final_quint[1]) + ',' + str(rt_final_quint[2]) + ']',
                          fontsize=title_fs)
    axes2[1, 1].set_title('Final RT: (' + str(rt_final_quint[2]) + ',' + str(rt_final_quint[3]) + ']',
                          fontsize=title_fs)
    axes2[2, 0].set_title('Final RT: (' + str(rt_final_quint[3]) + ',' + str(rt_final_quint[4]) + ']',
                          fontsize=title_fs)
    axes2[2, 1].set_visible(False)

    # Set tick and axis labels.
    for x in range(0, 3):
        for y in range(0, 2):
            axes2[x, y].xaxis.set_tick_params(labelsize=ax_fs)
            axes2[x, y].yaxis.set_tick_params(labelsize=ax_fs)
            axes2[x, y].set_xlabel('Minutes Since Post', fontsize=label_fs)
            axes2[x, y].set_ylabel('RT Count', fontsize=label_fs)

    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig2.savefig('Rates\\rt_hist_quint.png', dpi=300)

    # Cumulative histogram plot for retweet count at each minute, plotted seperately for each quintile.
    fig3, axes3 = plt.subplots(3, 2, figsize=(10, 9))

    # Select all data between the qunitile bounds and for which the time between tweet creation and retweet is less than
    # 240 minutes. Plot as cumulative histogram with three-minute bins.
    all_rt['delta'].loc[(all_rt['rt_final'] > 0) & (all_rt['rt_final'] <= rt_final_quint[0]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=True, grid=False, color='blue', edgecolor='black', ax=axes3[0, 0])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[0]) & (all_rt['rt_final'] <= rt_final_quint[1]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=True, grid=False, color='orange', edgecolor='black', ax=axes3[0, 1])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[1]) & (all_rt['rt_final'] <= rt_final_quint[2]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=True, grid=False, color='green', edgecolor='black', ax=axes3[1, 0])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[2]) & (all_rt['rt_final'] <= rt_final_quint[3]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=True, grid=False, color='red', edgecolor='black', ax=axes3[1, 1])

    all_rt['delta'].loc[(all_rt['rt_final'] > rt_final_quint[3]) & (all_rt['rt_final'] <= rt_final_quint[4]) &
                        (all_rt['delta'] <= 240)]. \
        hist(bins=np.arange(0, 241, 3), cumulative=True, grid=False, color='purple', edgecolor='black', ax=axes3[2, 0])

    # Title cumulative histogram subplots with quintile ranges. Set final subplot to not plot.
    axes3[0, 0].set_title('Final RT: [1,' + str(rt_final_quint[0]) + ']', fontsize=title_fs)
    axes3[0, 1].set_title('Final RT: (' + str(rt_final_quint[0]) + ',' + str(rt_final_quint[1]) + ']',
                          fontsize=title_fs)
    axes3[1, 0].set_title('Final RT: (' + str(rt_final_quint[1]) + ',' + str(rt_final_quint[2]) + ']',
                          fontsize=title_fs)
    axes3[1, 1].set_title('Final RT: (' + str(rt_final_quint[2]) + ',' + str(rt_final_quint[3]) + ']',
                          fontsize=title_fs)
    axes3[2, 0].set_title('Final RT: (' + str(rt_final_quint[3]) + ',' + str(rt_final_quint[4]) + ']',
                          fontsize=title_fs)
    axes3[2, 1].set_visible(False)

    # Set tick and axis labels.
    for x in range(0, 3):
        for y in range(0, 2):
            axes3[x, y].xaxis.set_tick_params(labelsize=ax_fs)
            axes3[x, y].yaxis.set_tick_params(labelsize=ax_fs)
            axes3[x, y].set_xlabel('Minutes Since Post', fontsize=label_fs)
            axes3[x, y].set_ylabel('Cumulative RTs', fontsize=label_fs)

    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig3.savefig('Rates\\rt_hist_cum_quint.png', dpi=300)

    # Plot mean RT count at each minute after tweet posting (up to 240 min) for each quintile of the final retweet
    # distribution.
    fig4, axes4 = plt.subplots(figsize=(10, 9))
    rt_count_quint = all_rt_min.groupby([pd.qcut(all_rt_min['rt_final'], q=5), 'minute'])['rt_count'].mean(). \
        unstack('rt_final')
    rt_count_quint_cumsum = rt_count_quint.cumsum(axis=0)
    rt_count_quint_cumsum.plot(ax=axes4)
    axes4.set_xticks(np.arange(0, 241, 30))

    # Set tick and axis labels.
    axes4.xaxis.set_tick_params(labelsize=ax_fs)
    axes4.yaxis.set_tick_params(labelsize=ax_fs)
    axes4.set_xlabel('Minutes Since Post', fontsize=label_fs)
    axes4.set_ylabel('Cumulative RTs', fontsize=label_fs)
    axes4.legend(fontsize=label_fs)

    plt.tight_layout()

    # Show figure, if desired.
    if show is True:
        plt.show()

    # Save figure, if desired.
    if save is True:
        fig4.savefig('Rates\\rt_count_cum_quint.png', dpi=300)

    # Plot mean % of final tweet value for each minute after tweet posting (up to 240 min) for each quintile of the
    # final retweet distribution.
    fig5, axes5 = plt.subplots(figsize=(10, 9))
    rt_per_quint = all_rt_min.groupby([pd.qcut(all_rt_min['rt_final'], q=5), 'minute'])['rt_per'].mean(). \
        unstack('rt_final')
    rt_per_quint_cumsum = rt_per_quint.cumsum(axis=0)
    rt_per_quint_cumsum.plot(ax=axes5)
    axes5.set_xticks(np.arange(0, 241, 30))

    # Set tick and axis labels.
    axes5.xaxis.set_tick_params(labelsize=ax_fs)
    axes5.yaxis.set_tick_params(labelsize=ax_fs)
    axes5.set_xlabel('Minutes Since Post', fontsize=label_fs)
    axes5.set_ylabel('% of Final RT Value', fontsize=label_fs)
    axes5.legend(fontsize=label_fs)

    plt.tight_layout()

    # Show figure if desired.
    if show is True:
        plt.show()

    # Save figure if desired.
    if save is True:
        fig5.savefig('Rates\\rt_per_cum_quint.png', dpi=300)


def rt_rate_crosstab_plot(df, image_range, show=True, save=False):
    # This function takes a dataframe with calculated count statistics over set time periods (e.g 5m, 6h) and produces
    # plots for a defined number of images. For each image type, the function plots the median number of RTs at each of
    # the set time periods, plotted seperately for each image source (e.g. Local NWS). The resulting figure always
    # plots with two columns and as many rows as necessary to fit all the provided image types, one image type per
    # subplot. The user can choose whether to show and/or save the figure (default is to show and not save). If the user
    # chooses to save the plot, the function will save it to a "Rates" folder. If this folder does not exist, an error
    # will be raised.

    # Define unique sources.
    scope_affs = df['user-scope_aff'].unique()[:-1]
    scope_affs.sort()

    # Define time increments, since posting of tweet, that diffusion count and rates are calculated over.
    count_times = [5, 10, 15, 30, 60, 120, 240, 360]

    # Define the number of rows to plot by dividing the length of the image range by 2 (the number of columns) and
    # rounding up.
    col_count = 2
    row_count = math.ceil(len(image_range) / 2)

    # Create figure and axes objects, colors to use when plotting sources, and axis counters (x and y).
    fig, axes = plt.subplots(row_count, col_count, figsize=(10, 9))
    sa_colors = ['blue', 'red', 'green', 'orange', 'purple', 'blue', 'gray', 'green', 'purple']
    fs = 14
    ls = 12
    x = 0
    y = 0

    # Filter the dataframe to only include an image type posted by a source, for each image type in selected image range
    # and for each unique source.
    for image in image_range:
        for i in range(0, len(scope_affs)):
            crosstab = df.loc[(df['image-type'] == image) & (df['user-scope_aff'] == scope_affs[i])]

            # Calculate the median RT count for the filtered image/source crosstab dataset for each of the already
            # calculated RT count times (5m, 10m, 15m, 30m, 1h, 2h, 4h, 6h), but only if the filtered image/source
            # crosstab dataset has more than 10 entries; else, append a NaN value.
            cross_tab = []
            for time in count_times:
                if len(crosstab) < 10:
                    cross_tab.append(np.nan)
                else:
                    cross_tab.append(crosstab['diffusion-rt_count_' + str(time) + 'm'].median())

            # For each selected image type, plot the median RT count at each count time for each unique source, where
            # the sources are colored by their affiliation, with solid lines for national sources and dashed lines for
            # local sources.
            if scope_affs[i][:5] == 'Local':
                line, = axes[x, y].plot(count_times, cross_tab, c=sa_colors[i], label=scope_affs[i], ls='dashed', lw=2)
            else:
                line, = axes[x, y].plot(count_times, cross_tab, c=sa_colors[i], label=scope_affs[i], ls='solid', lw=2)

        # Set labels for the x and y axes, and title the subplot with the image type.
        axes[x, y].set_xlabel('Minutes since post', fontsize=ls)
        axes[x, y].set_ylabel('Cumulative retweets', fontsize=ls)
        axes[x, y].set_title(image, fontsize=fs)

        # Using the x and y axis counters, plot the image types in order, so that the images plot left to right on the
        # first row before moving on to the next row.
        if (y + 1) < len(axes[x, :]):
            y += 1
        else:
            y = 0
            x += 1

    # If number of images/subplots is odd, set the final subplot to not plot.
    if len(image_range) % 2 != 0:
        axes[x, y].set_visible(False)

    # Create a title and legend for the plot. Place the legend on the bottom of the plot and centered, adjusting the
    # subplots to make room.
    fig.suptitle('Median retweet diffusion over time for select image X source pairings with >= 10 tweets')
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=[0.5, 0], ncol=4, fontsize=ls)
    fig.subplots_adjust(bottom=0.15, left=0.05, right=0.95, hspace=0.3)

    # Save the figure, if desired.
    if save is True:
        fig.savefig('Rates\\rt_rate_crosstab.png', dpi=300)

    # Show the figure, if desired.
    if show is True:
        plt.show()


def basic_scatter(df_calc, df_final, show=True, save=False):
    # This function creates two scatter plots - one based on data from the "calculated" tweet dataframe which is not
    # filtered and includes all outliers, and the second based on the "final" tweet dataframe which is filtered. The
    # scatter plot has retweet count as the x-axis and reply count as the y-axis. The user can choose whether to show
    # or save the display (default is to show and not save).

    # Notes: If the user chooses to save the figure, it will be saved in a 'Statistics' folder. If this folder does not
    # exist, an error will be raised.

    # Create a figure with two subplots.
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5))

    # Plot the non-filtered dataset RT vs reply scatter plot.
    axes[0].scatter(df_calc['diffusion-rt_count'], df_calc['diffusion-reply_count'])
    axes[0].set_xlabel('Retweets', fontsize=14)
    axes[0].set_ylabel('Replies', fontsize=14)
    axes[0].set_title('All tweets', fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    # Plot the filtered dataset RT vs reply scatter plot.
    axes[1].scatter(df_final['diffusion-rt_count'], df_final['diffusion-reply_count'])
    axes[1].set_xlabel('Retweets', fontsize=14)
    axes[1].set_ylabel('Replies', fontsize=14)
    axes[1].set_title('Without outlier tweet', fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # Save figure, if desired.
    if save is True:
        fig.savefig('Statistics\\rt_reply_scatter.png')

    # Show figure, if desired.
    if show is True:
        plt.show()


def scatter_size(df):
    """
    Plots a retweet-reply scatter plot where the area of the points are proportional to the number of occurences of
        that retweet/reply combination, and where the axes use the "symlog" scale in order to show all of the data in
        a compact manor

    Parameters:
        df: A tweet dataframe with retweet and reply data (Pandas dataframe)
    """

    # Count the occurence of each unique retweet/reply count combination
    gb = df.groupby(['diffusion-rt_count', 'diffusion-reply_count'])['tweet-id_trunc']. \
        size().reset_index().rename(columns={'tweet-id_trunc': 'count'})

    # Create a scatter plot where the size of the scatter is proportional to the number of occurences of each point
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    c = '#69d'
    ax.scatter(gb['diffusion-rt_count'], gb['diffusion-reply_count'], c='#69d', edgecolor='black', linewidth=0.25,
               alpha=0.5, s=gb['count'] * 25)

    # Set the x and y scales to "symlog" in order to display zero values while maintaining the compactness of the log
    # display
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')

    # Set axis limits and labels
    ax.set_ylim(bottom=-1)
    ax.set_xlim(left=-1)
    ax.set_xlabel('Retweets', fontsize=14, labelpad=10)
    ax.set_ylabel('Replies', fontsize=14, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Display a grid
    ax.grid(alpha=0.5)

    # Manually set legend for size values.
    size = [1, 10, 100]
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[0]),
               markersize=math.sqrt(size[0] * 25)),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[1]),
               markersize=math.sqrt(size[1] * 25)),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='w', label=str(size[2]),
               markersize=math.sqrt(size[2] * 25))]
    ax.legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=[0.15, -0.12], ncol=4, borderpad=1.5,
              fontsize=14)
    fig.subplots_adjust(bottom=0.2, left=0.15, right=0.95)

    plt.show()


def scatter_kde(df):
    """
    Plots a retweet-reply scatter plot where the color of the points are based on a Gaussian kernel density estimate,
        and where the axes use the log scale in order to show the data in a compact manor

    Parameters:
        df: A tweet dataframe with retweet and reply data (Pandas dataframe)
    """

    # Define x (retweets), y (replies), and z (the Gaussian kernel density estimate of the distribution)
    x = df['diffusion-rt_count'].to_numpy()
    y = df['diffusion-reply_count'].to_numpy()
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the values so that higher values always display on top of smaller values
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # Plot the scatter, using a Seaborn colormap to display the kernel density estimate
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = ListedColormap(sns.color_palette("Greens").as_hex())
    sc = ax.scatter(x, y, c=z, s=50, cmap=cmap, edgecolor=None)

    # Set the x and y scales to "log"
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Set the x and y limits and labels
    ax.set_ylim(bottom=0.5)
    ax.set_xlim(left=0.5)
    ax.set_xlabel('Retweets', fontsize=14)
    ax.set_ylabel('Replies', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Create a title and colorbar for the figure
    ax.set_title('All tweets', fontsize=18)
    fig.colorbar(sc)

    plt.show()


def scatter_transparent(df):
    """
    Plots a retweet-reply scatter plot where every point is plotted with low transparency, so that points with many
        overlaps show up darker, and where the axes use the symlog scale in order to show all of the data in a compact
        manor

    Parameters:
        df: A tweet dataframe with retweet and reply data (Pandas dataframe)
    """

    # Plot the retweet-reply scatter, with a low alpha for each point
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df['diffusion-rt_count'], df['diffusion-reply_count'], c='green', alpha=0.1)

    # Set the x and y scales to "symlog" in order to show all data (including zero values) while maintaining the
    # compactness of a log scale
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')

    # Set axis limits and labels
    ax.set_ylim(bottom=-0.5)
    ax.set_xlim(left=-0.5)
    ax.set_xlabel('Retweets', fontsize=14)
    ax.set_ylabel('Replies', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set axis ticklabels to show as integers, not in scientific notation
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    # Set figure title
    ax.set_title('All tweets', fontsize=18)

    plt.show()

def timeseries_line(df, freq, gb, sum_metric, dates, show=True, save=False):
    # This function creates a plot that displays the tweet count and the retweet count over time for each unique element
    # in the user-provided gb (groupby) column (e.g. 'image-type' or 'user-scope_aff'). The function plots tweet count
    # and retweet count as either a count for each user-provided time bin, or a cumulative count (depending on the
    # user-provided sum_metric variable). The user can choose whether to show and/or save the figure (default is to show
    # and not save).

    # Notes: function assumes that tweet-created_at is not the index of the tweet dataframe. Frequency must be provided
    # in minutes as an integer. Dates should be input as a list of timezone-aware datetime objects which correspond to
    # the start-time of the plot, the end-time of the plot, and any significant dates inbetween that should be
    # highlighted. If the user chooses to save the plot, the figure will be saved in a "Timing" folder. If this folder
    # does not exist, an error will be raised.

    # Set the dataframe index to the created at column.
    df = df.set_index('tweet-created_at')

    # Format frequency as a string (for groupby) and convert to a string with the number of hours for the title.
    freq_str = str(freq) + 'Min'
    freq_title = str('{:0.0f}'.format(freq / 60)) + 'h'

    # Create simplified groupby titles (for use in the title later).
    if gb == 'image-type':
        gb_title = 'image'
    elif gb == 'user-scope_aff':
        gb_title = 'source'
    else:
        gb_title = input('Please input a simplified title description for the groupby column you"ve provided')

    # Group the tweet dataframe into time bins (defined by the user-provided frequency) and by the user-provided groupby
    # column and calculate the tweet count and retweet sum for each group. Fill NaN values with zeros.
    count = df.groupby([pd.Grouper(freq=freq_str), gb]).count()['diffusion-rt_count'].unstack()
    rt_sum = df.groupby([pd.Grouper(freq=freq_str), gb]).sum()['diffusion-rt_count'].unstack()
    count = count.fillna(0)
    rt_sum = rt_sum.fillna(0)

    # Calculate cumulative counts and retweet sums for each time bin.
    count_cumsum = count.cumsum()
    rt_cumsum = rt_sum.cumsum()

    # Create a figure and define plotting variables.
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    td = timedelta(hours=(freq / 120 - 5))
    lw = 2
    fs = 14
    lp = 10
    ls = 12

    # Plot the tweet and RT counts or cumulative tweet and RT counts, depending on the user-provided summary metric.
    pd.plotting.register_matplotlib_converters()
    if sum_metric == 'count':
        for col in count.columns:
            axes[0].plot(count.index + td, count[col], linewidth=lw, label=col)
        for col in rt_sum.columns:
            axes[1].plot(rt_sum.index + td, rt_sum[col], linewidth=lw, label=col)
        axes[0].set_ylabel('Tweet Count', fontsize=fs, labelpad=lp)
        axes[1].set_ylabel('Total RT', fontsize=fs, labelpad=lp)

    elif sum_metric == 'cumsum':
        for col in count.columns:
            axes[0].plot(count_cumsum.index + td, count_cumsum[col], linewidth=lw, label=col)
        for col in rt_sum.columns:
            axes[1].plot(rt_cumsum.index + td, rt_cumsum[col], linewidth=lw, label=col)
        axes[0].set_ylabel('Cumulative Tweet Count', fontsize=fs, labelpad=lp)
        axes[1].set_ylabel('Cumulative RT', fontsize=fs, labelpad=lp)

    # Timing variables
    start = min(dates)
    end = max(dates)
    date_plot = [x for x in dates if x != max(dates) and x != min(dates)]

    # Adjust axes and labels.
    for ax in axes:
        ax.set_xlim(start, end)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(None)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=ls)
        ax.tick_params(axis='y', labelsize=ls)
        ax.legend(loc='upper left', fontsize=ls)

        # Add light gray lines at times of significance.
        for date in date_plot:
            ax.axvline(date, color='gray', alpha=0.5)

    plt.tight_layout()

    if show is True:
        plt.show()

    if save is True:
        fig.savefig('Timing\\timeseries_' + gb_title + '_' + sum_metric + '_' + freq_title + '_line.png', dpi=300)


def rt_quint_comp(df, stat, show=True, save=False):
    # From a dataframe with calculated RT diffusion count and rate statistics for multiple fixed-interval timeframes
    # (e.g. 5m, 6h), create three plots. The first plots the median RT count/rate (determined by user-provided stat) for
    # each quintile of the final RT distribution. The second and third plots compare the median count/rates within a
    # quintile to each other - first, comparing to the minimum count/rate (5m/6h respectively); second, to the maximum
    # count/rate(6h/5m). User can choose whether to save or show the output (default is to show but not save)

    # Notes: stat must be formatted as a string - 'count' for RT count, 'rate' for RT rate. If input does not match one
    # of these two options, an error will be raised. If user chooses to save the output, it will save to a 'RT Rates'
    # folder. If this folder does not exist, an error will be raised.

    # Collect RT count columns, if user-provided stat is 'count'.
    stat_cols = []
    if stat == 'count':
        for col in df.columns:
            if col[:19] == 'diffusion-rt_count_':
                stat_cols.append(col)
        min_label = '5m'
        max_label = '6h'

    # Collect RT rate columns, if user-provided stat is 'rate'.
    elif stat == 'rate':
        for col in df.columns:
            if col[:17] == 'diffusion-rt_rate':
                stat_cols.append(col)
        min_label = '6h'
        max_label = '5m'

    # If user-provided stat is not 'count' or 'rate', collect no columns.
    else:
        stat_cols.append(None)
        min_label = None
        max_label = None

    # Calculate median count/rate statistics for each quintile of the final RT distribution.
    quint_gb = df.groupby([pd.qcut(df['diffusion-rt_count'], q=5)])[stat_cols].median()

    # Transpose the count/rate groupby so that each row corresponds to a count/rate statistic.
    quint_gb = quint_gb.T

    # Plot a bar graph where the median count/rate statistic is calculated for each data quintile.
    fig, ax = plt.subplots()
    quint_gb.plot(kind='bar', ax=ax)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels=['5m', '10m', '15m', '30m', '1h', '2h', '4h', '6h'])
    ax.tick_params(axis='both', labelsize=12)

    if save is True:
        fig.savefig('RT Rates\\rt_' + stat + '_med_quint.png')

    if show is True:
        plt.show()

    # Compare each count/rate statistic to the max and min rate/count. Perform for each data quintile.
    quint_comp_min = pd.DataFrame()
    quint_comp_max = pd.DataFrame()

    # Iterates over each column of the groupby, obtaining the column name (label) and content (values).
    for label, content in quint_gb.items():
        # Divide by the minimum of the column (this is the 5m count and 6h rate).
        quint_comp_min[label] = content / content.min()

    # Rename the quintile ranges.
    quint_comp_min.columns = ['q1', 'q2', 'q3', 'q4', 'q5']

    # Plot the count/rate minimum comparisons for each rate statistic for each quintile.
    fig2, ax2 = plt.subplots()
    quint_comp_min.plot(kind='bar', ax=ax2)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels=[5, 10, 15, 30, 60, 120, 240, 360])
    ax2.tick_params(axis='both', labelsize=12)

    if save is True:
        fig2.savefig('RT Rates\\rt_' + stat + '_med_quint_' + min_label + '_comp.png')

    if show is True:
        plt.show()

    # Iterates over each column of the groupby, obtaining the column name (label) and content (values).
    for label, content in quint_gb.items():
        # Divide by the maximum of the column (this is the 6h count and the 5m rate).
        quint_comp_max[label] = content / content.max()

    # Rename the quintile ranges.
    quint_comp_max.columns = ['q1', 'q2', 'q3', 'q4', 'q5']

    # Plot the count/rate maximum comparions for each rate statistic for each quintile.
    fig3, ax3 = plt.subplots()
    quint_comp_max.plot(kind='bar', ax=ax3)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels=[5, 10, 15, 30, 60, 120, 240, 360])
    ax3.tick_params(axis='both', labelsize=12)

    if show is True:
        plt.show()

    if save is True:
        fig3.savefig('RT Rates\\rt_' + stat + '_med_quint_' + max_label + '_comp.png')

# </editor-fold>


# <editor-fold desc="Other">
def url_display(df):
    # Select an image dataset and display all URLs in browser.
    url = df['tweet-url']
    locs = list(range(0, len(url)))
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s --incognito'
    for i in locs:
        webbrowser.get(chrome_path).open_new_tab(url.iloc[i])
# </editor-fold>
