import pandas as pd
import twitter_toolkit as ttk
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import rgb2hex
import numpy as np
import math
import random
import os


pd.options.mode.chained_assignment = None

# Input user-provided column order
col_order = pd.read_csv('col_order.csv')['New Order'].tolist()

# Define start and end times
timezone = pytz.timezone('UTC')
harvey_start = timezone.localize(datetime(2017, 8, 17, 0))
harvey_end = timezone.localize(datetime(2017, 9, 2, 15))


# Read/calculate data
def readdata(how, order):
    """
    Reads in a Twitter database to be filtered and used for data analysis

    Parameters:
        how: Determines how to input data (string). 'calc' will create the database from scratch, merging datasets
                 together and calculating diffusion metrics. 'read' reads in a previously created database, saving time
        order: Order for columns to be arranged in (list of column names as strings)
    """

    if how == 'calc':
        # Merge original HIMN data and missing data together
        tweets = ttk.merge(him_all_file='HIM Data\\himn_116k_simple.csv',
                           himn_coded_file='Harvey\\Data\\himn_coded.csv',
                           old_missing_all_file='Harvey\\Data\\tweets_old_missing.csv',
                           old_missing_fore_file='Harvey\\Data\\tweet_data_missing_forecast.csv',
                           new_missing_file='Harvey\\Data\\new_missing_coded.csv',
                           orig_file='HIM Data\\him_origs_coded.csv',
                           start=harvey_start,
                           end=harvey_end)

        # Calculate diffusion counts, overall and over time
        tweets = ttk.diff_calc_basic(tweets, 'Harvey\\Data\\harvey_tweet_diffusion_files')
        tweets = ttk.tweet_diffusion_calc(tweets, 'Harvey\\Data', 'harvey_tweet_diffusion_files')

        # Merge mesoscale discussion coding with tweet data file
        md_codes = pd.read_csv('Harvey\\Data\\md_coded.csv')[['tweet-id_trunc', 'image-type_meso-disc_spc',
                                                              'image-type_meso-disc_wpc']]
        md_codes['tweet-id_trunc'] = md_codes['tweet-id_trunc'].astype(str)
        tweets['tweet-id_trunc'] = tweets['tweet-id_trunc'].astype(str)
        tweets = pd.merge(tweets, md_codes, on='tweet-id_trunc', how='outer')
        tweets['image-type_meso-disc_spc'] = tweets['image-type_meso-disc_spc'].map({'yes': 1, 'no': 0})
        tweets['image-type_meso-disc_wpc'] = tweets['image-type_meso-disc_wpc'].map({'yes': 1, 'no': 0})

        # Clean data
        for col in tweets.columns:
            if col[:7] == 'Unnamed':
                tweets.drop(col, axis=1, inplace=True)

        tweets = tweets.reset_index().reindex(columns=order)
        tweets.set_index('tweet-id', inplace=True)

        # Fill numeric columns nans with zeros
        numeric_columns = tweets.select_dtypes(include=['number']).columns
        tweets[numeric_columns] = tweets[numeric_columns].fillna(0)

        # Output
        tweets.to_csv('HIM Data\\tweets_HIM.csv')

    elif how == 'read':
        # Read in data
        tweets = pd.read_csv('HIM Data\\tweets_HIM.csv')

    else:
        tweets = pd.DataFrame()

    return tweets


tweets_all = readdata('read', col_order)

# <editor-fold desc="Dates">
timezone = pytz.timezone('US/Central')
harvey_start = timezone.localize(datetime(2017, 8, 17, 0))
harvey_end = timezone.localize(datetime(2017, 9, 2, 0))
harvey_cone = timezone.localize(datetime(2017, 8, 23, 10))
harvey_landfall = timezone.localize(datetime(2017, 8, 25, 22, 0))
harvey_dates = [harvey_start, harvey_end, harvey_cone, harvey_landfall]
harvey_labels = ['Hurricane watch\nissued for\nTexas coast', 'First U.S.\nHarvey landfall\n']
# </editor-fold>

# <editor-fold desc="Images">
image_types = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
               'River Flood Forecast', 'Convective Outlook/Forecast', 'Rainfall Outlook/Forecast', 'Cone', 'Text',
               'Tropical Outlook', 'Watch/Warning (Exp)', 'Watch/Warning']
images_display = ['Watch/Warning (Exp)', 'Watch/Warning', 'Cone', 'Rainfall Outlook/Forecast', 'River Flood Forecast',
                  'Convective Outlook/Forecast', 'Model Output', 'Text', 'Tropical Outlook', 'Key Messages',
                  'Other - Forecast', 'Other - Non-Forecast', 'Multiple']
#images_tl = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output', ''
#             'Rainfall Outlook/Forecast', 'River Flood Forecast', 'Convective Outlook/Forecast', 'Cone',
#             'Watch/Warning (Exp)', 'Watch/Warning']
images_tl_no_ww = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
                   'Rainfall Forecast/Outlook', 'River Flood Forecast', 'SPC Convective Products', 'Cone']
images_tl_no_conv = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
                     'Rainfall Forecast/Outlook', 'River Flood Forecast', 'Cone']
#images = images_tl[::-1]
images_no_ww = images_tl_no_ww[::-1]
images_no_conv = images_tl_no_conv[::-1]

images_ww_map = ['Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW',
                 'Non-WW', 'WW (Exp)', 'WW (Non-Exp)']
images_ww_dict = dict(zip(image_types, images_ww_map))

image_cols = ['trop-out', 'cone', 'arrival', 'prob', 'surge', 'key-msg', 'ww', 'threat-impact', 'conv-out',
                  'meso-disc', 'rain-fore', 'rain-out', 'riv-flood', 'spag', 'text', 'model', 'evac',
                  'other-fore', 'other-non-fore', 'video']
ww_cols = ['ww_exp', 'ww_cone', 'ww_md']
other_cols = ['official', 'unofficial', 'spanish', 'english']

image_filter_cols = ['image-type_multi', 'image-type_other-non-fore', 'image-type_other-fore', 'image-type_key-msg',
                  'image-type_model', 'image-type_riv-flood', 'image-type_conv', 'image-type_rain',
                  'image-type_cone', 'image-type_text', 'image-type_trop-out', 'image-type_ww_exp', 'image-type_ww']
all_cols = image_cols + ww_cols + other_cols

prefix = 'image-type_'
img_col_names = [prefix + col for col in image_cols]
ww_col_names = [prefix + col for col in ww_cols]
other_col_names = ['image-branding_off', 'image-branding_unoff', 'image-lang_spanish', 'image-lang_english']
all_col_names = img_col_names + ww_col_names + other_col_names
# </editor-fold>

# <editor-fold desc="Sources">
sources_all = ['Local NWS', 'Local NWS (Exp)', 'Local Non-NWS Wx Gov', 'Local EM', 'Local Other Gov',
               'Local News Media', 'Local Wx Media', 'Local Other Wx', 'Local Other - Non-Wx', 'National NWS',
               'National NWS (Exp)', 'National Non-NWS Wx Gov', 'National EM', 'National Other Gov',
               'National News Media', 'National Wx Media', 'National Other Wx', 'National Other Non-Wx']
sources_exp = ['National NWS (Exp)', 'Local NWS (Exp)']
sources_all_noexp = [source for source in sources_all if source not in sources_exp]

sources_filter_noexp = ['Local NWS', 'Local Non-NWS Wx Gov', 'Local EM', 'Local Other Gov', 'Local News Media',
                        'Local Wx Media', 'Local Other Wx', 'Local Other - Non-Wx', 'National NWS', 'National Wx Media']
sources = ['National NWS (Exp)', 'National NWS', 'National Wx Media', 'Local NWS (Exp)', 'Local NWS',
           'Local Wx Media', 'Local News Media', 'Local Non-NWS Government', 'Local Wx Bloggers']
sources_no_ww = ['National NWS', 'National Wx Media', 'Local NWS', 'Local Wx Media',
                 'Local News Media', 'Local Non-NWS Government', 'Local Wx Bloggers']
sources_local = ['Local NWS', 'Local Wx Media', 'Local News Media', 'Local Non-NWS Government', 'Local Wx Bloggers']
sources_tl = sources[::-1]
sources_tl_no_ww = sources_no_ww[::-1]
# </editor-fold>

# <editor-fold desc="Other values">
save_cols = ['tweet-created_at', 'tweet-url', 'tweet-text', 'user-screen_name', 'user-followers', 'user-scope_aff',
             'image-type', 'image-branding_off', 'image-branding_unoff', 'diffusion-rt_count', 'diffusion-qt_count',
             'diffusion-reply_count']

lang_brands = ['English', 'Spanish', 'NWS', 'Non-NWS']
lang_brand_cols = ['image-lang_english', 'image-lang_spanish', 'image-branding_off', 'image-branding_unoff']
lang_brand_values = [1]

# </editor-fold>

# <editor-fold desc="Filter data and output">
tweets_harvey = ttk.tweet_filter(tweets_all, filters=['time', 'source', 'risk', 'relevant', 'forecast'])

tweets_harvey = ttk.scope_aff_filter(tweets_harvey, col_order=col_order, sep_exp=True)
tweets_harvey = ttk.image_filter(tweets_harvey)

# Convert the tweet created at column to datetime format.
tweets_harvey['tweet-created_at'] = pd.to_datetime(tweets_harvey['tweet-created_at'], format='%Y-%m-%d %H:%M:%S%z')
tweets_harvey['tweet-created_at'] = tweets_harvey['tweet-created_at'].dt.tz_convert('US/Central')

# Output coded, merged, cleaned dataset
#tweets_harvey.to_csv('tweets_harvey.csv')

# </editor-fold>

qual1_start = timezone.localize(datetime(2017, 8, 23, 9))
qual1_end = timezone.localize(datetime(2017, 8, 23, 12))
qual2_start = timezone.localize(datetime(2017, 8, 24, 9))
qual2_end = timezone.localize(datetime(2017, 8, 24, 12))
qual3_start = timezone.localize(datetime(2017, 8, 27, 9))
qual3_end = timezone.localize(datetime(2017, 8, 27, 12))

tweets_qual1 = ttk.tweet_filter(tweets_harvey, date_range=[qual1_start, qual1_end])
tweets_qual2 = ttk.tweet_filter(tweets_harvey, date_range=[qual2_start, qual2_end])
tweets_qual3 = ttk.tweet_filter(tweets_harvey, date_range=[qual3_start, qual3_end])

images_noexp = [image for image in images_display if image != 'Watch/Warning (Exp)']
tweets_noexp = tweets_harvey.loc[tweets_harvey['image-type'].isin(images_noexp)]
#ttk.rate_plot(tweets_noexp, gb='user-scope_aff', metric='rt', title='source')

#tweets_qual1 = tweets_qual1.loc[tweets_qual1['user-screen_name'] == 'nwssanantonio']
#tweets_qual1[['tweet-created_at', 'tweet-text', 'tweet-url', 'media-type', 'user-screen_name', 'user-followers',
#              'user-scope_aff', 'image-type', 'diffusion-rt_count',
#              'diffusion-reply_count']].to_csv('tweets_harvey_qual1.csv')
#ttk.url_display(tweets_qual1.sort_values('diffusion-rt_count', ascending=False))

#tweets_qual2 = tweets_qual2.loc[tweets_qual2['user-screen_name'] == 'jdharden']
#tweets_qual2[['tweet-created_at', 'tweet-text', 'tweet-url', 'media-type', 'user-screen_name', 'user-followers',
#              'user-scope_aff', 'image-type', 'diffusion-rt_count',
#              'diffusion-reply_count']].to_csv('tweets_harvey_qual2.csv')
#ttk.url_display(tweets_qual2)
#ttk.url_display(tweets_qual2.sort_values('diffusion-rt_count', ascending=False))

#tweets_qual3 = tweets_qual3.loc[tweets_qual3['user-screen_name'] == 'jimcantore']
#tweets_qual3[['tweet-created_at', 'tweet-text', 'tweet-url', 'media-type', 'user-screen_name', 'user-followers',
#              'user-scope_aff', 'image-type', 'diffusion-rt_count',
#              'diffusion-reply_count']].to_csv('tweets_harvey_qual3.csv')
#ttk.url_display(tweets_qual3)
#ttk.url_display(tweets_qual3.sort_values('diffusion-rt_count', ascending=False))

# Obtain RGBA color values from the Tab 10 color scheme (the Matplotlib default).
colors = plt.cm.tab10(np.linspace(0, 1, 9))

# Convert the RGBA values to RGB, then to hex. Append the first and second hex colors twice to account for the
# twelve image categories.
color_hex = []
for color in colors:
    rgb = color[:3]
    color_hex.append(rgb2hex(rgb))
color_hex.append(color_hex[0])
color_hex.append(color_hex[1])
color_hex.append(color_hex[2])
color_hex.append(color_hex[3])
color_hex.append(color_hex[4])

fig, ax = plt.subplots()
colors = dict(zip(sources, color_hex))
for i, image in enumerate(images_display):
    source_qual1 = tweets_qual1.loc[tweets_qual1['image-type'] == image]
    ax.scatter(source_qual1['tweet-created_at'], source_qual1['diffusion-rt_count'], s=source_qual1['diffusion-rt_count'], label=image, c=color_hex[i])
    #ax.plot(source_qual1['tweet-created_at'], source_qual1['diffusion-rt_count'], label=source, c=color_hex[i])
#plt.scatter(tweets_qual1['tweet-created_at'], tweets_qual1['diffusion-rt_count'], s=50, labels=sources, c=tweets_qual1['user-scope_aff'].apply(lambda x: colors[x]))
#ax.set_yscale('symlog')
#tweets_qual1[['tweet-created_at', 'diffusion-rt_count', 'user-screen_name']].apply(lambda row: ax.text(*row), axis=1)
#ax.set_yscale('symlog')
plt.legend()
plt.close()
#plt.show()

df_images = ttk.descr_stats(tweets_harvey, all_col_names, [1], all_col_names, ['rt', 'reply'])
df_images.sort_values('Tweet Count', inplace=True)
#print(df_images)

tweets_harvey.loc[(tweets_harvey['image-branding_off'] == 1) & (tweets_harvey['image-branding_unoff'] == 0), 'image-branding'] = 'Official'
tweets_harvey.loc[(tweets_harvey['image-branding_off'] == 0) & (tweets_harvey['image-branding_unoff'] == 1), 'image-branding'] = 'Unofficial'
#tweets_harvey.loc[(tweets_harvey['image-branding_off'] == 1) & (tweets_harvey['image-branding_unoff'] == 1), 'image-branding'] = 'Both'
#tweets_harvey.loc[(tweets_harvey['image-branding_off'] == 0) & (tweets_harvey['image-branding_unoff'] == 0), 'image-branding'] = 'Neither'
tweets_harvey['image-ww_map'] = tweets_harvey['image-type'].map(images_ww_dict)
tweets_harvey['image-branding_X_ww'] = tweets_harvey['image-branding'] + ' ' + tweets_harvey['image-ww_map']
images_noexp = [image for image in images_display if image != 'Watch/Warning (Exp)']
tweets_noexp = tweets_harvey.loc[tweets_harvey['image-type'].isin(images_noexp)]

# <editor-fold desc="Figures, tables, and calculations">
# Figure 1 - Sankey flow chart
#ttk.sankey(tweets_all, labels=False, show=True)

# Table 1 - Diffusion detail (W/W Exp, W/W Non-Exp, Non-W/W)
df_descr_exp = ttk.descr_stats(tweets_harvey, ['image-ww_map'], ['WW (Exp)', 'WW (Non-Exp)', 'Non-WW'],
                               ['WW (Exp)', 'WW (Non-Exp)', 'Non-WW'], ['rt', 'reply'])
#print('Table 1 - Diffusion detail (W/W Exp, W/W Non-Exp, Non-W/W)')
#print(df_descr_exp)

# Figure 3 - Scatter (size)
#ttk.scatter_size(tweets_harvey, show=False)

# Figure 4a - timeseries (W/W Exp, W/W Non-Exp, Non-W/W) with outliers and medians
#ttk.timeseries_ww_wwexp_nonww(tweets_harvey, freq=360, dates=harvey_dates, median=True, show=True)

# Filter to remove outliers
tweets_harvey = ttk.tweet_filter(tweets_harvey, rt_range=np.arange(0, 10000))

# Figure 4 - timeseries (W/W Exp, W/W Non-Exp, Non-W/W), no outliers, only totals
#ttk.timeseries_ww_wwexp_nonww(tweets_harvey, freq=360, dates=harvey_dates, median=False, show=True)

# Figure 5a - rate plot (W/W exp, W/W non-exp, non-W/W)
#ttk.rate_plot(tweets_harvey, gb='image-ww_map', metric='rt', title='watch/warning content')

# Figure 5b - rate plot (official vs unofficial)
#ttk.rate_plot(tweets_noexp, gb='image-branding', metric='rt', title='image branding')

# Figure 5c - rate plot (branding X watch/warning content)
#ttk.rate_plot(tweets_noexp, gb='image-branding_X_ww', metric='rt', title='image branding X watch/warning content')

# Figure 5d - rate plot (source)
#ttk.rate_plot(tweets_noexp, gb='user-scope_aff', metric='rt', title='source')

# Figure 5e - rate plot (images)
#ttk.rate_plot(tweets_noexp, gb='image-type', metric='rt', title='image type')

# Table 2 - source descriptive table
df_descr_source = ttk.descr_stats(tweets_harvey, ['user-scope_aff'], sources, sources, ['rt', 'reply'])
df_descr_source.sort_values('Median rt', ascending=False, inplace=True)
#print('Table 2 - Source descriptive table')
#print(df_descr_source[['Accounts', 'Tweet Count', 'Median rt', 'Median reply']])

# Figure 6 - Source timeline
#ttk.timeline(tweets_harvey, ['user-scope_aff'], sources[::-1], sources[::-1], 'diffusion-rt_count', 'user-agency_ind',
#             harvey_dates)

# Table 3 - image descriptive table
df_descr_image = ttk.descr_stats(tweets_harvey, ['image-type'], images_display, images_display, ['rt', 'reply'])
df_descr_image.sort_values('Median rt', ascending=False, inplace=True)
#print('Table 3 - Image descriptive table')
#print(df_descr_image[['Accounts', 'Tweet Count', 'Median rt', 'Median reply']])

# Table 4a - other image code descriptive table
df_descr_other = ttk.descr_stats(tweets_harvey, other_col_names, [1], other_col_names, ['rt', 'reply'])
#print('Table 4a - Other image codes descriptive table')
#print(df_descr_other)

# Table 4b - other image code descriptive table, without experimental warning graphics
df_descr_other_noexp = ttk.descr_stats(tweets_noexp, other_col_names, [1], other_col_names, ['rt', 'reply'])
#print('Table 4b - Other image codes descriptive table, without experimental warning graphics')
#print(df_descr_other_noexp)

# Figure 7 - image timeline
image_tl = ttk.cat_midpoint(tweets_harvey, 'image-type', 'diffusion-rt_count', True)
#ttk.timeline(tweets_harvey, ['image-type'], image_tl[::-1], image_tl[::-1], 'diffusion-rt_count', 'image-branding_off',
#             harvey_dates)

qual1_images = tweets_qual1['image-type'].drop_duplicates().tolist()
qual2_images = tweets_qual2['image-type'].drop_duplicates().tolist()
qual3_images = tweets_qual3['image-type'].drop_duplicates().tolist()
qual1_images_sort = [image for image in image_tl if image in qual1_images]
qual2_images_sort = [image for image in image_tl if image in qual2_images]
qual3_images_sort = [image for image in image_tl if image in qual3_images]
qual1_dates = [qual1_start, qual1_end]
qual2_dates = [qual2_start, qual2_end]
qual3_dates = [qual3_start, qual3_end]
ttk.timeline(tweets_harvey, ['image-type'], qual3_images_sort[::-1], qual3_images_sort[::-1], 'diffusion-rt_count', 'user-scope_loc',
             qual3_dates)
end

# Source retweet Mann-Whitney U calculation
pvals_source = ttk.mannwhitneyu_test(tweets_harvey, 'user-scope_aff', 'matrix', 'diffusion-rt_count')
pvals_source.to_csv('pval_mwu_source_rt.csv')

# Source reply Mann-Whitney U calculation
pvals_source = ttk.mannwhitneyu_test(tweets_harvey, 'user-scope_aff', 'matrix', 'diffusion-reply_count')
pvals_source.to_csv('pval_mwu_source_reply.csv')

# Image retweet Mann-Whitney U calculation
pvals_image = ttk.mannwhitneyu_test(tweets_harvey, 'image-type', 'matrix', 'diffusion-rt_count')
pvals_image.to_csv('pval_mwu_image_rt.csv')

# Image reply Mann-Whitney U calculation
pvals_image = ttk.mannwhitneyu_test(tweets_harvey, 'image-type', 'matrix', 'diffusion-reply_count')
pvals_image.to_csv('pval_mwu_image_reply.csv')

tweets_noexp['image-ww_map'] = tweets_noexp['image-type'].map(images_ww_dict)
tweets_noexp['ww_brand'] = tweets_noexp['image-branding_off'].astype(str) + tweets_noexp['image-ww_map']
df_descr_ww_brand = ttk.descr_stats(tweets_noexp, ['ww_brand'],
                                    ['1.0WW (Non-Exp)', '1.0Non-WW', '0.0WW (Non-Exp)', '0.0Non-WW'],
                                    ['Official W/W', 'Official Non-W/W', 'Unofficial W/W', 'Unofficial Non-W/W'],
                                    ['rt', 'reply'])
print(df_descr_ww_brand[['Accounts', 'Tweet Count', 'Median rt', 'Median reply']])
#print(tweets_noexp.groupby('ww_brand').count()['tweet-id_trunc'])

# </editor-fold>

end

tweets_harvey_mult = ttk.tweet_filter(tweets_harvey, image_range=['Multiple'])
tweets_mult_gb = tweets_harvey_mult.groupby(image_filter_cols[1:] + ['media-url_num']).size().reset_index().rename(columns={0: 'count'})
tweets_mult_gb['Images'] = (tweets_mult_gb[image_filter_cols[1:]] == 1).apply(lambda x: ','.join(tweets_mult_gb[image_filter_cols[1:]].columns[x]), axis=1)
for a, b in dict(zip(image_filter_cols, image_types)).items():
    tweets_mult_gb['Images'] = tweets_mult_gb['Images'].str.replace(a, b)
tweets_mult_gb['# of Images'] = tweets_mult_gb['Images'].str.count(',') + 1
tweets_mult_gb = tweets_mult_gb[['Images', 'media-url_num', '# of Images', 'count']]
#tweets_mult_gb['Images'] = tweets_mult_gb['Images'].str.replace(dict(zip(image_filter_cols, image_types)))
tweets_mult_gb.sort_values('count', ascending=False, inplace=True)
print(tweets_mult_gb)
#tweets_mult_gb.to_csv('tweets_mult.csv')





end

prefix = 'image-type_'
img_col_names = [prefix + col for col in image_cols]
ww_col_names = [prefix + col for col in ww_cols]
other_col_names = ['image-branding_off', 'image-branding_unoff', 'image-lang_spanish', 'image-lang_english']
all_col_names = img_col_names + ww_col_names + other_col_names

#ttk.timeseries_ww_wwexp_nonww(tweets_harvey, 240, harvey_dates, save=True)
