import irma_tools as itools
import twitter_toolkit as ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

pd.options.mode.chained_assignment = None
plt.rcParams['font.family'] = 'Arial'

# Read in and filter data
irma_tweets = itools.readdata(how='read', save=True)
irma_original = irma_tweets[irma_tweets['referenced_tweets1.type'] == 'original_tweet']
irma_rel = irma_original[irma_original['relevant'] == 1]

print(irma_rel[['created_at', 'created_at_date6h', 'hour_3h', 'date_range', 'date', 'hour']])

# <editor-fold desc="Defaults and conversions">
# Data to display value dictionaries
scope_aff_vals = irma_rel.groupby('user.scope_aff').agg({'created_at': 'count', 'retweet_count': 'median'}). \
    sort_values('retweet_count').reset_index()['user.scope_aff']
media_dict = {'animated_gif': 'GIF', 'video': 'Video', 'multi_photo': 'Multi-Photo', 'photo': 'Photo',
              'text-only': 'Text-Only'}
bot_dict = {'Yes': 'Bot Tweet', 'No': 'Non-Bot Tweet'}
hazard_dict = {'surge': 'Surge', 'multiple': 'Multiple', 'tc': 'TC', 'rain_flood': 'Rain/Flood',
               'convective': 'Convective', 'other': 'Other'}
risk_dict = {'multiple': 'Multiple', 'past': 'Past', 'non_ww_fore': 'Forecast', 'obs': 'Observational',
             'ww_fore': 'Watch/Warning', 'other': 'Other'}

# Palette dictionaries
# scope_palette = ['#0072f0', '#00b6cb', '#f10096']
hazard_palette = {'surge': '#66a61e', 'multiple': '#7570b3', 'tc': '#1b9e77', 'convective': '#d95f02',
                  'rain_flood': '#e7298a', 'other': 'lightgray'}
media_palette = {'animated_gif': '#f66d00', 'video': '#ffa800', 'multi_photo': '#f10096', 'photo': '#0072f0',
                 'text-only': '#00b6cb'}
risk_palette = {'multiple': '#ccebc5', 'past': '#decbe4', 'non_ww_fore': '#fbb4ae', 'obs': '#b3cde3',
                'ww_fore': '#fed9a6', 'other': 'lightgray'}
bot_palette = {'No': '#8dd3c7', 'Yes': '#bebada'}

# Define start and end times
timezone = pytz.timezone('UTC')
tz = pytz.timezone('US/Eastern')
irma_start = tz.localize(datetime(2017, 8, 30, 0))
irma_end = tz.localize(datetime(2017, 9, 13, 0))
irma_dates = [irma_start, irma_end]
fake_dates = [tz.localize(datetime(2017, 8, 31, 23)), tz.localize(datetime(2017, 9, 2, 2))]
dates = [date.strftime('%m-%d') for date in pd.date_range(irma_start, irma_end)]
hours = np.arange(0, 25)

# Define engagement variables
engages = ['retweet', 'reply', 'quote', 'like']
# </editor-fold>

# <editor-fold desc="Simplified columns">
# Gather date/time dummy code columns - note, with new dummy code creation in Itools, prefix for all categories below
#     will just be the column name
date_cols = sorted(['date_' + str(d) for d in irma_tweets['date'].drop_duplicates().tolist()])
hour_cols = sorted(['hour_' + str(h) for h in irma_tweets['hour'].drop_duplicates().tolist()])
date6h_cols = sorted(['date6h_' + str(dh) for dh in irma_tweets['created_at_date6h'].drop_duplicates().tolist()])
hour3h_cols = sorted(['hour3h_' + str(h3) for h3 in irma_tweets['hour_3h'].drop_duplicates().tolist()])
drange_cols = sorted(['drange_' + str(dr) for dr in irma_tweets['date_range'].drop_duplicates().tolist()])

# Create simplified version of Irma tweets for processing in Google Data Studio
metacols = ['source', 'referenced_tweets1.type', 'media.type', 'includes.media', 'media_tweet-type']
timecols = ['created_at', 'created_at_date6h', 'hour_3h', 'date_range'] + date_cols + date6h_cols + drange_cols + \
            hour_cols + hour3h_cols
diffcols = ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'retweet_log_norm', 'follower_log_norm',
            'all_source_retweetZ', 'rel_source_retweetZ', 'all_source_engageZ', 'rel_source_engageZ']
usercols = ['user.followers_count', 'user.username', 'user.scope', 'user.local_scope', 'user.agency',
            'user.affiliation', 'user.scope_aff']
codecols = ['deleted_qt', 'relevant', 'spanish', 'bot_tweet', 'bot_media']
hazcols = ['hazard', 'hazard_tc', 'hazard_surge', 'hazard_rain_flood', 'hazard_convective', 'hazard_mult',
           'hazard_other', 'hazard_non_rel', 'hazard_join']
riskcols = ['risk', 'risk_non_ww_fore', 'risk_ww_fore', 'risk_obs', 'risk_past', 'risk_mult', 'risk_other',
            'risk_non_rel', 'risk_join']
othercols = ['hazard_risk']

simp_cols = metacols + timecols + diffcols + usercols + codecols + hazcols + riskcols + othercols

# Create very simplified version of Irma tweets for processing in R
basic_cols = ['id', 'media.type', 'user.username', 'user.followers_count', 'retweet_count', 'date', 'date_range', 'hour_3h',
              'bot_tweet', 'hazard', 'risk']
# </editor-fold>

# <editor-fold desc="Additional segmentation and printing">
irma_orig_media = irma_original[irma_original['includes.media'] == 'media']
irma_non_rel = irma_tweets[irma_tweets['relevant'] == 0]
irma_rel_bot = irma_rel.loc[irma_rel['bot_tweet'] == 1]
irma_rel_nobot = irma_rel.loc[irma_rel['bot_tweet'] != 1]
irma_rel_to_code = irma_rel.loc[pd.isna(irma_rel['hazard_tc'])]
irma_spanish = irma_tweets[irma_tweets['spanish'] == 'yes']
irma_deletedqt = irma_tweets[irma_tweets['deleted_qt'] == 'yes']
irma_rel_url = irma_rel.loc[irma_rel['includes.url'] == 1]
irma_rel_hashtag = irma_rel.loc[irma_rel['includes.hashtag'] == 1]
irma_rel_mention = irma_rel.loc[irma_rel['includes.mention'] == 1]
irma_rel_media = irma_rel.loc[irma_rel['includes.media'] == 'media']
irma_rel_exclam = irma_rel.loc[irma_rel['!'] == 1]
irma_rel_question = irma_rel.loc[irma_rel['?'] == 1]
irma_rel_cap = irma_rel.loc[irma_rel['includes.CAP'] == 1]
print('All Tweets : ' + str(len(irma_tweets)))
print('Original Tweets: ' + str(len(irma_original)))
#print('Original Media Tweets: ' + str(len(irma_orig_media)))
print('Relevant: ' + str(len(irma_rel)))
#print('Relevant Bot Tweets: ' + str(len(irma_rel_bot)))
#print('Relevant Non-Bot Tweets: ' + str(len(irma_rel_nobot)))
#print('Relevant Tweets (to be Coded): ' + str(len(irma_rel_to_code)))
print('Relevant Media Tweets: ' + str(len(irma_rel_media)))
# print('Unique Users Using Media: ' + str(len(irma_rel_media['user.username'].drop_duplicates())))
# print('Relevant Tweets with a URL: ' + str(len(irma_rel_url)))
# print('Unique Users Using URLs: ' + str(len(irma_rel_url['user.username'].drop_duplicates())))
# print('Relevant Tweets with a Hashtag: ' + str(len(irma_rel_hashtag)))
# print('Unique Users Using Hashtags: ' + str(len(irma_rel_hashtag['user.username'].drop_duplicates())))
# print('Relevant Tweets with a Mention: ' + str(len(irma_rel_mention)))
# print('Unique Users using Mentions: ' + str(len(irma_rel_mention['user.username'].drop_duplicates())))
# print('Relevant Tweets with a !: ' + str(len(irma_rel_exclam)))
# print('Unique Users using !: ' + str(len(irma_rel_exclam['user.username'].drop_duplicates())))
# print('Relevant Tweets with a ?: ' + str(len(irma_rel_question)))
# print('Unique Users using ?: ' + str(len(irma_rel_question['user.username'].drop_duplicates())))
# print('Relevant Tweets with ALL CAPS: ' + str(len(irma_rel_cap)))
# print('Unique Users using ALL CAPS: ' + str(len(irma_rel_cap['user.username'].drop_duplicates())))
# print('Spanish: ' + str(len(irma_spanish)))
# print('Deleted QT: ' + str(len(irma_deletedqt)))
# </editor-fold>

# <editor-fold desc="Old visualizations">
# ttk.sankey(irma_tweets, filters=['time', 'original', 'relevant'], user_col='user.username', labels=False)

# fig, ax = plt.subplots(figsize=(11, 8.5))
# sns.scatterplot(data=irma_rel, x='follower_log_norm', y='retweet_log_norm', hue='user.scope', style='user.affiliation', ax=ax)
# ax.axvline(x=4.5, c='gray')
# ax.axvline(x=5.5, c='gray')
# fig.savefig('user_scatter.png', dpi=300, bbox_inches='tight')
# plt.tight_layout()
# plt.show()

# ttk.timeseries_ww_wwexp_nonww(df=irma_rel,
#                               freq=6*60,
#                               column='user.bot',
#                               values=irma_rel['user.bot'].drop_duplicates().tolist(),
#                               id_field='tweet-url',
#                               date_field='created_at',
#                               diff_field='retweet_count',
#                               colors=user_bot_colors,
#                               dates=irma_dates,
#                               median=False,
#                               save=True)
#
# itools.count_diff(irma_rel, 'Relevant Tweets', 'hour_3h'
#            , date_filter=True, date_start=tz.localize(datetime(2017, 8, 30)), date_end=tz.localize(datetime(2017, 9, 13))
#            , split='bot_tweet'
#            )
# <editor-fold>

# # <editor-fold desc="User, media, hazard, and risk timelines (currently inactive)">
# ttk.timeline(df=irma_rel,
#              value_cols=['user.scope_aff'],
#              values=scope_aff_vals,
#              labels=scope_aff_vals,
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
#
# ttk.timeline(df=it_rel,
#              value_cols=['user.scope_aff'],
#              values=scope_aff_vals,
#              labels=scope_aff_vals,
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
#
# ttk.timeline(df=irma_rel,
#              value_cols=['media.type'],
#              values=list(media_dict.keys())[::-1],
#              labels=list(media_dict.values())[::-1],
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
#
# ttk.timeline(df=irma_rel,
#              value_cols=['hazard'],
#              values=list(hazard_dict.keys())[::-1],
#              labels=list(hazard_dict.values())[::-1],
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
#
# ttk.timeline(df=irma_rel,
#              value_cols=['risk'],
#              values=list(risk_dict.keys())[::-1],
#              labels=list(risk_dict.values())[::-1],
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
# # </editor-fold> (cu #

# <editor-fold desc="Selecting tweets for image coding ICR (currently inactive)">
# Filter to remove non-media tweets, bot tweets, "other" tweets, and past tweets
# irma_ic = irma_rel[(irma_rel['hazard'] != 'other') & (irma_rel['risk'] != 'past') &
#                    (irma_rel['bot_tweet'] != 1) & (irma_rel['includes.media'] == 'media')]
# print(len(irma_ic))

# Select 82 tweets for the first round of ICR coding for image content
# irma_ic_samp = irma_ic.sample(n=82).sort_values('created_at')
# irma_ic_samp['round'] = 1
# code_cols = ['round', 'id', 'tweet-url', 'created_at', 'user.username', 'text', 'media.count', 'media.type',
#              'media1.url', 'media2.url', 'media3.url', 'media4.url']
# irma_ic_samp = irma_ic_samp[code_cols]
# print(len(irma_ic_samp))
# irma_ic_samp.to_csv('irma_ic_samp.csv')

# Select 200 tweets (that haven't already been selected) for second round of ICR coding for image content
# irma_ic_samp1 = pd.read_csv('Irma\\Data\\Image Coding\\irma_ic_samp.csv')
# irma_ic_samp1['id'] = irma_ic_samp1['tweet-url'].str[-18:]
# ids1 = irma_ic_samp1['id'].tolist()
#
# irma_ic2 = irma_ic[~irma_ic['id'].isin(ids1)]
# irma_ic_samp2 = irma_ic2.sample(n=200).sort_values('created_at')
# irma_ic_samp2['round'] = 2
# code_cols = ['round', 'id', 'tweet-url', 'created_at', 'user.username', 'text', 'media.count', 'media.type',
#              'media1.url', 'media2.url', 'media3.url', 'media4.url']
# irma_ic_samp2 = irma_ic_samp2[code_cols]
# print(len(irma_ic_samp2))
# irma_ic_samp2.to_csv('Irma\\Data\\Image Coding\\irma_ic_samp2.csv')
# </editor-fold>

# irma_rel_basic = irma_rel[basic_cols]
# irma_rel_basic.to_csv('Twitter_R\\irma_rel_basic.csv')


