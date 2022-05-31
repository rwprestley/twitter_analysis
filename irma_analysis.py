import irma_tools as itools
import twitter_toolkit as ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import seaborn as sns

pd.options.mode.chained_assignment = None
plt.rcParams['font.family'] = 'Arial'

# Read in and filter data
irma_tweets = itools.readdata(how='read', save=True)
irma_original = irma_tweets[irma_tweets['referenced_tweets1.type'] == 'original_tweet']
irma_rel = irma_original[irma_original['relevant'] == 1]
irma_rel = itools.diff_calcs(irma_rel, 'rel', 'retweet')

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
image_dict = {'image_type.two': 'TWO', 'image_type.cone': 'Cone', 'image_type.arrival': 'Arrival',
              'image_type.prob': 'Probability', 'image_type.spag': 'Spaghetti', 'image_type.wind': 'Wind Map',
              'image_type.surge': 'Surge', 'image_type.info': 'Infographic', 'image_type.ww': 'Watch/Warning',
              'image_type.ti': 'Threat/Impact', 'image_type.conv': 'Convective', 'image_type.rain': 'Rainfall/Flooding',
              'image_type.sat': 'Satellite', 'image_type.radar': 'Radar', 'image_type.adv': 'Advisory',
              'image_type.env': 'Env Cue', 'image_type.text': 'Text', 'image_type.model': 'Model Output',
              'image_type.orisk': 'Other Risk', 'image_type.onon': 'Other Non-Risk'}
brand_dict = {'nws': 'NWS/NOAA', 'non_nws': 'Non-NWS/NOAA', 'no_branding': 'No Branding', 'multiple': 'Multiple'}

# Palette dictionaries
# scope_palette = ['#0072f0', '#00b6cb', '#f10096']
hazard_palette = {'surge': '#66a61e', 'multiple': '#7570b3', 'tc': '#1b9e77', 'convective': '#d95f02',
                  'rain_flood': '#e7298a', 'other': 'lightgray'}
media_palette = {'animated_gif': '#f66d00', 'video': '#ffa800', 'multi_photo': '#f10096', 'photo': '#0072f0',
                 'text-only': '#00b6cb'}
risk_palette = {'multiple': '#ccebc5', 'past': '#decbe4', 'non_ww_fore': '#fbb4ae', 'obs': '#b3cde3',
                'ww_fore': '#fed9a6', 'other': 'lightgray'}
bot_palette1 = {'No': '#8dd3c7', 'Yes': '#bebada'}
bot_palette2 = {'No': 'lightskyblue', 'Yes': 'white'}
image_palette = {'image_type.two': 'blue', 'image_type.cone': 'blue', 'image_type.arrival': 'blue',
                 'image_type.prob': 'blue', 'image_type.spag': 'blue', 'image_type.wind': 'blue',
                 'image_type.surge': 'blue', 'image_type.info': 'blue', 'image_type.ww': 'blue',
                 'image_type.ti': 'blue', 'image_type.conv': 'blue', 'image_type.rain': 'blue',
                 'image_type.sat': 'blue', 'image_type.radar': 'blue', 'image_type.adv': 'blue',
                 'image_type.env': 'blue', 'image_type.text': 'blue', 'image_type.model': 'blue',
                 'image_type.orisk': 'blue', 'image_type.onon': 'blue'}
brand_palette = {'nws': '#603f83', 'non_nws': '#e9738d', 'multiple': 'goldenrod', 'no_branding': 'white'}

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
date6h_cols = sorted(['created_at_date6h_' + str(dh) for dh in irma_tweets['created_at_date6h'].drop_duplicates().
                     tolist()])
hour3h_cols = sorted(['hour_3h_' + str(h3) for h3 in irma_tweets['hour_3h'].drop_duplicates().tolist()])
drange_cols = sorted(['date_range_' + str(dr) for dr in irma_tweets['date_range'].drop_duplicates().tolist()])

# Create simplified version of Irma tweets for processing in Google Data Studio
metacols = ['source', 'referenced_tweets1.type', 'media.type', 'includes.media', 'media_tweet-type']
timecols = ['created_at', 'created_at_date6h', 'hour_3h', 'date_range', 'date', 'hour'] + date_cols + date6h_cols + \
           drange_cols + hour_cols + hour3h_cols
diffcols = ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'retweet_log_norm', 'follower_log_norm',
            'all_source_retweetZ', 'rel_source_retweetZ', 'all_source_engageZ', 'rel_source_engageZ']
usercols = ['user.followers_count', 'user.username', 'user.scope', 'user.local_scope', 'user.agency',
            'user.affiliation', 'user.scope_aff']
codecols = ['deleted_qt', 'relevant', 'spanish', 'bot_tweet', 'bot_media']
hazcols = ['hazard', 'hazard_tc', 'hazard_surge', 'hazard_rain_flood', 'hazard_convective', 'hazard_mult',
           'hazard_other', 'hazard_non_rel', 'hazard_join']
riskcols = ['risk', 'risk_non_ww_fore', 'risk_ww_fore', 'risk_obs', 'risk_past', 'risk_mult', 'risk_other',
            'risk_non_rel', 'risk_join']
imagecols = list(image_dict.keys()) + list(brand_dict.keys()) + ['image.type', 'image.brand', 'image_join',
                                                                 'brand_join', 'image.brand_type']
othercols = ['id', 'hazard_risk']

simpcols = metacols + timecols + diffcols + usercols + codecols + hazcols + riskcols + imagecols + othercols

# Create very simplified version of Irma tweets for processing in R
basic_cols = ['id', 'media.type', 'user.username', 'user.followers_count', 'retweet_count', 'date', 'date_range',
              'hour_3h', 'bot_tweet', 'hazard', 'risk'] + list(image_dict.keys())
# </editor-fold>

# <editor-fold desc="Additional segmentation and printing">
irma_orig_media = irma_original[irma_original['includes.media'] == 'media']
irma_non_rel = irma_tweets[irma_tweets['relevant'] == 0]
irma_rel_bot = irma_rel.loc[irma_rel['bot_tweet'] == 1]
irma_rel_nobot = irma_rel.loc[irma_rel['bot_tweet'] != 'Yes']
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
irma_time1 = irma_rel.loc[irma_rel['created_at'] <= tz.localize(datetime(2017, 9, 5))]
print('All Tweets : ' + str(len(irma_tweets)))
print('Original Tweets: ' + str(len(irma_original)))
# print('Original Media Tweets: ' + str(len(irma_orig_media)))
print('Relevant: ' + str(len(irma_rel)))
# print('Relevant Bot Tweets: ' + str(len(irma_rel_bot)))
# print('Relevant Non-Bot Tweets: ' + str(len(irma_rel_nobot)))
# print('Relevant Tweets (to be Coded): ' + str(len(irma_rel_to_code)))
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
print('Irma Time 1: ' + str(len(irma_time1)))
# </editor-fold>

# <editor-fold desc="Old visualizations">
# -------------------------------------------------------------------------------------------------------------------- #
# Sankey filter plot for Irma data
# -------------------------------------------------------------------------------------------------------------------- #
# ttk.sankey(irma_tweets, filters=['time', 'original', 'relevant'], user_col='user.username', labels=False)

# -------------------------------------------------------------------------------------------------------------------- #
# Retweet-reply scatter plot for each user, coded by their scope (Local or National)
# -------------------------------------------------------------------------------------------------------------------- #
# irma_rel_user = irma_rel.groupby(['user.username', 'user.scope']).agg(
#     {'retweet_count': 'median', 'reply_count': 'median'}).reset_index()
# print(irma_rel_user)
# fig, ax = plt.subplots(figsize=(8, 8))
# irma_rel_local = irma_rel_user[irma_rel_user['user.scope'] != 'National']
# irma_rel_nat = irma_rel_user[irma_rel_user['user.scope'] == 'National']
# ax.scatter(irma_rel_local['retweet_count'], irma_rel_local['reply_count'], color='#2CAE66', edgecolor='gray',
#            label='local')
# ax.scatter(irma_rel_nat['retweet_count'], irma_rel_nat['reply_count'], color='#FFA177', edgecolor='gray',
#            label='national')
# ax.set_ylabel('Median Replies', fontsize=14, labelpad=10)
# ax.set_xlabel('Median Retweets', fontsize=14, labelpad=10)
# ax.set_xlim(xmax=400)
# ax.legend(fontsize=14)
# plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Follower-median RT scatter plot for each tweet, coded by user characteristics (e.g. user strip plots)
# -------------------------------------------------------------------------------------------------------------------- #
# fig, ax = plt.subplots(figsize=(11, 8.5))
# sns.scatterplot(data=irma_rel, x='follower_log_norm', y='retweet_log_norm', hue='user.scope', style='user.affiliation',
#                 ax=ax)
# ax.axvline(x=4.5, c='gray')
# ax.axvline(x=5.5, c='gray')
# fig.savefig('user_scatter.png', dpi=300, bbox_inches='tight')
# plt.tight_layout()
# plt.show()

# Counts and median RTs by 3-hour time bin bar plot, stratified by bot tweets
# itools.count_diff(irma_rel, 'Relevant Tweets', 'hour_3h',
#                   date_filter=True, date_start=tz.localize(datetime(2017, 8, 30)),
#                   date_end=tz.localize(datetime(2017, 9, 13)), split='bot_tweet')

# -------------------------------------------------------------------------------------------------------------------- #
# Hazard and risk overlap counts and mean source-adjusted diffusion table
# -------------------------------------------------------------------------------------------------------------------- #
# name = []
# hazlist = []
# risklist = []
# count = []
# rtZ = []
# for hazard in irma_rel_nobot['hazard'].drop_duplicates().tolist():
#     for risk in irma_rel_nobot['risk'].drop_duplicates().tolist():
#         name.append(hazard + '_' + risk)
#         hazlist.append(hazard)
#         risklist.append(risk)
#         count.append(len(irma_rel_nobot.loc[(irma_rel_nobot['hazard'] == hazard) & (irma_rel_nobot['risk'] == risk)]))
#         rtZ.append(irma_rel_nobot.loc[(irma_rel_nobot['hazard'] == hazard) & (irma_rel_nobot['risk'] == risk)][
#                        'rel_source_retweetZ'].mean())
#
# haz_risk_gb = pd.DataFrame()
# haz_risk_gb['name'] = name
# haz_risk_gb['hazard'] = hazlist
# haz_risk_gb['risk'] = risklist
# haz_risk_gb['count'] = count
# haz_risk_gb['rtZ'] = rtZ
# haz_risk_gb.sort_values('count', ascending=False, inplace=True)
# haz_risk_gb = haz_risk_gb.loc[haz_risk_gb['count'] >= 25]
# print(haz_risk_gb)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare diffusion spread among hazard/risk overlaps to make an assessment of which factor (hazard, risk) is more
# important for predicting diffusion
# -------------------------------------------------------------------------------------------------------------------- #
# haz_max = []
# haz_min = []
# for haz in haz_risk_gb['hazard'].drop_duplicates().tolist():
#     haz_max.append(haz_risk_gb.loc[haz_risk_gb['hazard'] == haz]['rtZ'].max())
#     haz_min.append(haz_risk_gb.loc[haz_risk_gb['hazard'] == haz]['rtZ'].min())
#
# risk_max = []
# risk_min = []
# for rsk in haz_risk_gb['risk'].drop_duplicates().tolist():
#     risk_max.append(haz_risk_gb.loc[haz_risk_gb['risk'] == rsk]['rtZ'].max())
#     risk_min.append(haz_risk_gb.loc[haz_risk_gb['risk'] == rsk]['rtZ'].min())
#
# hazdf = pd.DataFrame()
# hazdf['hazard'] = haz_risk_gb['hazard'].drop_duplicates().tolist()
# hazdf['max'] = haz_max
# hazdf['min'] = haz_min
# hazdf['range'] = hazdf['max'] - hazdf['min']
# print(hazdf)
# print(hazdf.loc[hazdf['hazard'] != 'other']['range'].mean())
#
# riskdf = pd.DataFrame()
# riskdf['risk'] = haz_risk_gb['risk'].drop_duplicates().tolist()
# riskdf['max'] = risk_max
# riskdf['min'] = risk_min
# riskdf['range'] = riskdf['max'] - riskdf['min']
# print(riskdf)
# print(riskdf.loc[(riskdf['risk'] != 'past') & (riskdf['risk'] != 'other')]['range'].mean())
#
# fig1, ax1 = plt.subplots()
# sns.barplot(x='hazard', y='rtZ', hue='risk', data=haz_risk_gb, palette='Set2', ax=ax1)
# ax1.legend(loc='lower center', ncol=2, fontsize=20)
# ax1.tick_params(axis='both', labelsize=20)
# ax1.set_xlabel('hazard', fontsize=20)
# ax1.set_ylabel('RT Score', fontsize=20)
# plt.show()
#
# fig2, ax2 = plt.subplots()
# sns.barplot(x='risk', y='rtZ', hue='hazard', data=haz_risk_gb, palette='Set2', ax=ax2)
# ax2.legend(loc='lower center', ncol=2, fontsize=20)
# ax2.tick_params(axis='both', labelsize=20)
# ax2.set_xlabel('risk', fontsize=20)
# ax2.set_ylabel('RT Score', fontsize=20)
# plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Plot different diffusion statistics for different categorical variables
# -------------------------------------------------------------------------------------------------------------------- #
# data_cols = ['media.type', 'bot_tweet', 'hazard', 'risk']
# value_dicts = [media_dict, bot_dict, hazard_dict, risk_dict]
# palettes = [media_palette, bot_palette1, hazard_palette, risk_palette]
# long_names = ['Media Type', 'Bot Tweet', 'Hazard', 'Risk Information']
# short_names = ['media', 'bot', 'haz', 'risk']
# diff_cols = ['source_retweetZ', 'source_raw_retweet_diff', 'source_raw_retweet_per', 'follower_retweet']
# diff_aggs = ['Z', 'Diff', 'Per', 'Foll']
# agg_stat = ['mean', 'median', 'median', 'median']
#
# for i, var in enumerate(data_cols):
#     for diff_agg in diff_aggs:
#         itools.diff_bar(irma_rel, 'rel', 'retweet', var, diff_agg, value_dicts[i], palettes[i], long_names[i],
#                         short_names[i], show=True, save=False)

# -------------------------------------------------------------------------------------------------------------------- #
# Calculate distribution characteristics (skewness & kurtosis) for various diffusion statistics (retweet and total
#    engagements) and permutations (e.g. log-normalized, for all relevent vs for all original tweet, etc.) for each
#    user
# -------------------------------------------------------------------------------------------------------------------- #
# dfs = [irma_rel, irma_rel, irma_original, irma_rel_nobot]
# refs = ['rel', 'rel', 'original', 'nonbot']
# diffs = ['retweet', 'engage']
# diff_types = ['count', 'log_norm', 'log_norm', 'log_norm']
# user_ordered = irma_rel.groupby('user.username')['retweet_count'].sum().reset_index(). \
#     sort_values('retweet_count', ascending=False)['user.username'].tolist()
#
# user_rel_count, user_nonbot_count = ([] for i in range(2))
# user_dict = {}
# for user in user_ordered:
#     user_vals = {}
#     for engage in diffs:
#         for i, df in enumerate(dfs):
#             skew = df[df['user.username'] == user][engage + '_' + diff_types[i]].skew()
#             kurt = df[df['user.username'] == user][engage + '_' + diff_types[i]].kurt()
#
#             user_vals[refs[i] + '_count'] = len(df[df['user.username'] == user])
#             user_vals[engage + '_rel_med'] = dfs[0][dfs[0]['user.username'] == user][engage + '_count'].median()
#             user_vals[refs[i] + '_' + engage + '_' + diff_types[i] + '_skew'] = skew
#             user_vals[refs[i] + '_' + engage + '_' + diff_types[i] + '_kurt'] = kurt
#
#     user_dict[user] = user_vals
# user_df = pd.DataFrame.from_dict(user_dict, orient='index').sort_values('retweet_rel_med', ascending=False)
# print(user_df)
# user_df.to_csv('user_dist.csv')

# -------------------------------------------------------------------------------------------------------------------- #
# Plot retweet distributions for each user individually
# -------------------------------------------------------------------------------------------------------------------- #
# for user in user_ordered:
#     sns.histplot(data=irma_rel[irma_rel['user.username'] == user],
#                  x='retweet_log_norm', kde=True).set(xlabel=None, ylabel=None, title=user + ': relevant tweets')
#     plt.savefig('Irma\\Visualizations\\User Distributions\\' + user + '.png', dpi=300)
#     plt.show()


# -------------------------------------------------------------------------------------------------------------------- #
# Plot retweet distributions for each user together in one graphic
# -------------------------------------------------------------------------------------------------------------------- #
# Define subplots and set initial x and y values
# fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(11, 8.5))
# x = 0
# y = 0
#
# # For each user...
# for user in user_ordered:
#     # Plot the retweet distribution
#     sns.histplot(data=irma_rel[irma_rel['user.username'] == user],
#                  x='retweet_plus1', kde=True, log_scale=True, ax=axes[x, y]).set(xlabel=None, ylabel=None)
#
#     # Label the subplot with the username
#     axes[x, y].text(0.5, 0.8, user, ha='center', transform=axes[x, y].transAxes)
#
#     # Remove ticklabels for all but the left-most and bottom-most subplots
#     if y > 0:
#         axes[x, y].yaxis.set_ticklabels([])
#
#     if x < 7:
#         axes[x, y].xaxis.set_ticklabels([])
#
#     # Set the axes of each subplot over the same range and domain
#     axes[x, y].set_xlim(xmax=10**4)
#     axes[x, y].set_ylim(ymax=150)
#
#     # Move through the subplots from left to right and from top to bottom
#     y += 1
#     if y >= 6:
#         x += 1
#         y = 0
#
# # Plot the total distribution in the final subplot
# sns.histplot(data=irma_rel, x='retweet_plus1', bins=20, kde=True, log_scale=True, ax=axes[7, 5]).set(xlabel=None,
#                                                                                                      ylabel=None)
#
# # Save and show
# plt.tight_layout()
# plt.savefig('Irma\\Visualizations\\User Distributions\\user_rt_log_hist.png')
# plt.show()

# <editor-fold>

# # <editor-fold desc="User, media, hazard, and risk timelines (currently inactive)">
# ttk.timeline(df=irma_rel,
#              value_cols=['user.scope_aff'],
#              values=scope_aff_vals,
#              labels=scope_aff_vals,
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              color_val='Yes',
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
#              color_val='Yes'
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
#              color_val='Yes',
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
#              color_val='Yes',
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
#              color_val='Yes',
#              dates=irma_dates,
#              datetime_col='created_at',
#              save=False)
# # </editor-fold>

# # <editor-fold desc="Selecting tweets for image coding ICR (currently inactive)">
# # Filter to remove non-media tweets, bot tweets, "other" tweets, and past tweets
# irma_ic = irma_rel[(irma_rel['hazard'] != 'other') & (irma_rel['risk'] != 'past') &
#                    (irma_rel['bot_tweet'] != 1) & (irma_rel['includes.media'] == 'media')]
# print(len(irma_ic))
#
# # Select 82 tweets for the first round of ICR coding for image content
# # irma_ic_samp = irma_ic.sample(n=82).sort_values('created_at')
# # irma_ic_samp['round'] = 1
# # code_cols = ['round', 'id', 'tweet-url', 'created_at', 'user.username', 'text', 'media.count', 'media.type',
# #              'media1.url', 'media2.url', 'media3.url', 'media4.url']
# # irma_ic_samp = irma_ic_samp[code_cols]
# # print(len(irma_ic_samp))
# # irma_ic_samp.to_csv('irma_ic_samp.csv')
#
# # Select 200 tweets (that haven't already been selected) for second round of ICR coding for image content
# irma_ic_samp1 = pd.read_csv('Irma\\Data\\Image Coding\\irma_ic_samp.csv')
# irma_ic_samp1['id'] = irma_ic_samp1['tweet-url'].str[-18:]
# ids1 = irma_ic_samp1['id'].tolist()
# #
# # irma_ic2 = irma_ic[~irma_ic['id'].isin(ids1)]
# # irma_ic_samp2 = irma_ic2.sample(n=200).sort_values('created_at')
# # irma_ic_samp2['round'] = 2
# # code_cols = ['round', 'id', 'tweet-url', 'created_at', 'user.username', 'text', 'media.count', 'media.type',
# #              'media1.url', 'media2.url', 'media3.url', 'media4.url']
# # irma_ic_samp2 = irma_ic_samp2[code_cols]
# # print(len(irma_ic_samp2))
# # irma_ic_samp2.to_csv('Irma\\Data\\Image Coding\\irma_ic_samp2.csv')
#
# # Select the remaining tweets (that weren't selected for ICR coding) for full coding for image content
# irma_ic_samp2 = pd.read_csv('Irma\\Data\\Image Coding\\irma_ic_samp2.csv')
# irma_ic_samp2['id'] = irma_ic_samp2['tweet-url'].str[-18:]
# ids2 = irma_ic_samp2['id'].tolist()
# ids_all = ids1 + ids2
#
# irma_ic_tocode = irma_ic[~irma_ic['id'].isin(ids_all)]
# print(irma_ic_tocode)
# print(len(irma_ic_tocode))
# irma_ic_tocode.to_csv('Irma\\Data\\Image Coding\\irma_ic_tocode.csv')

# </editor-fold>

# <editor-fold desc="Creating simplified datasets for analysis elsewhere (currently inactive)"
# irma_rel_basic = irma_rel[basic_cols]
# irma_rel_basic.to_csv('Twitter_R\\irma_rel_basic.csv')

# irma_rel_simp = irma_rel[simpcols]
# irma_rel_simp.set_index('id', inplace=True)
# irma_rel_simp.to_csv('irma_rel_simp.csv')
# # </editor-fold>

# <editor-fold desc="Grouping users based on their station affiliation (currently inactive)"
# User groups
user_nws = ['nwstampabay', 'nwsmiami', 'nhc_atlantic', 'nws', 'nwsspc', 'nwswpc', 'nhc_surge']
user_twc = ['weatherchannel', 'jimcantore']
user_miami_wsvn7 = ['7weather', 'philferro7', 'viviangonzalez7', 'bcameron7', 'karlenechavis']
user_miami_nbc6 = ['johnmoralesnbc6', 'ryannbc6', 'adambergnbc6', 'angienbc6', 'stevemacnbc6']
user_miami_cbs4 = ['lissettecbs4', 'craigsetzer', 'davewarrencbs4', 'lizhortontv']
user_miami_wplg10 = ['juliedurda', 'lukedorris', 'jenniferwxwoman', 'thebettydavis']
user_tampa_bn9 = ['bn9weather', 'mcclurewx', 'mike_clay']
user_tampa_wtsp10 = ['grantwtsp', 'bobbywtsp', 'rickearbeywtsp', 'ashleybatey']
user_tampa_fox13 = ['paulfox13', 'weatherlindsay', 'fox13tyler', 'jimweberfox', 'daveofox13']
user_tampa_abc = ['gregdeeweather', 'denisphillipswx', 'jasonadamswfts', 'tampabayweather']
user_tampa_nbc8 = ['wkrged', 'tampabaysjulie', 'wflaleigh', 'sjervewfla', 'wflaian']
user_groups = [user_nws, user_twc, user_miami_wsvn7, user_miami_nbc6, user_miami_cbs4, user_miami_wplg10,
               user_tampa_bn9, user_tampa_wtsp10, user_tampa_fox13, user_tampa_abc, user_tampa_nbc8]
user_titles = ['NWS (National & Local)', 'The Weather Channel', 'Miami: WSVN FOX 7', 'Miami: WTVJ NBC 6',
               'Miami: WFOR CBS 4', 'Miami: WPLG ABC 10', 'Tampa: Spectrum Bay News 9', 'Tampa: WTSP CBS 10',
               'Tampa: WTVT FOX 13', 'Tampa: WFTS ABC 28', 'Tampa: WFLA NBC 8']
save_name = ['nws', 'twc', 'miami_fox7', 'miami_nbc6', 'miami_cbs4', 'miami_abc10', 'tampa_bn9', 'tampa_cbs10',
             'tampa_fox13', 'tampa_abc28', 'tampa_nbc8']

for group, name in zip(user_groups, save_name):
    for user in group:
        irma_rel.loc[irma_rel['user.username'] == user, 'user_group'] = name

# print(irma_rel.groupby('user_group').agg({'user.username': 'nunique', 'tweet-url': 'count', 'retweet_count': 'median'}))
# # </editor-fold>

# <editor-fold desc="Image code analysis (currently inactive)"
# -------------------------------------------------------------------------------------------------------------------- #
# Basic descriptives, original coded columns - image type
# -------------------------------------------------------------------------------------------------------------------- #
# image_type, user_count, tweet_count, median_rt, median_rtZ = ([] for i in range(5))
# for col in list(image_dict.keys()):
#     image_type.append(col)
#     tweet_count.append(len(irma_time1.loc[irma_time1[col] == 1]))
#     user_count.append(len(irma_time1.loc[irma_time1[col] == 1]['user.username'].drop_duplicates().tolist()))
#     median_rt.append(irma_time1.loc[irma_time1[col] == 1]['retweet_count'].median())
#     median_rtZ.append(irma_time1.loc[irma_time1[col] == 1]['rel_source_retweetZ'].mean())
#
# img_desc = pd.DataFrame(list(zip(image_type, tweet_count, user_count, median_rt, median_rtZ)),
#                         columns=['type', 'count', 'user_count', 'medRT', 'meanRTz']).sort_values('count',
#                                                                                                  ascending=False)
# img_desc.to_csv('img_desc_orig.csv')
# print(img_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# # Basic descriptives, original coded columns - image branding
# -------------------------------------------------------------------------------------------------------------------- #
# image_type, user_count, tweet_count, median_rt, median_rtZ = ([] for i in range(5))
# for col in list(brand_dict.keys()):
#     image_type.append(col)
#     tweet_count.append(len(irma_rel.loc[irma_rel[col] == 1]))
#     user_count.append(len(irma_rel.loc[irma_rel[col] == 1]['user.username'].drop_duplicates().tolist()))
#     median_rt.append(irma_rel.loc[irma_rel[col] == 1]['retweet_count'].median())
#     median_rtZ.append(irma_rel.loc[irma_rel[col] == 1]['rel_source_retweetZ'].mean())
#
# img_desc = pd.DataFrame(list(zip(image_type, tweet_count, user_count, median_rt, median_rtZ)),
#                         columns=['type', 'count', 'user_count', 'medRT', 'meanRTz']).sort_values('meanRTz',
#                                                                                                  ascending=False)
# img_desc.to_csv('brand_desc_orig.csv')
# # print(img_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# # Original coded columns - RT Z-score bar plot
# -------------------------------------------------------------------------------------------------------------------- #
# fig, ax = plt.subplots(figsize=(8.5, 11))
# img_desc = img_desc.sort_values('meanRTz')
# img_desc.plot(kind='barh', y='meanRTz', x='type', ax=ax, edgecolor='black')
# labels = [image_dict[k] for k in img_desc['type'].tolist()]
# ax.set_yticklabels(labels)
# ax.axvline(0, color='black')
# ax.set_title('Image Type (Original Coded Columns) - Mean Source-Adjusted RT Scores')
# # plt.savefig('image_orig_rtZ.png', dpi=300, bbox_inches='tight')
# # plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# # Basic descriptives, image type - mutually exclusive columns
# -------------------------------------------------------------------------------------------------------------------- #
# itype_excl_desc = irma_rel.groupby('image.type').agg({'image.type': 'count', 'user.username': 'nunique',
#                                                       'retweet_count': 'median', 'rel_source_retweetZ': 'mean'})\
#     .rename(columns={'image.type': 'count', 'user.username': 'user_count', 'retweet_count': 'medRT',
#                      'rel_source_retweetZ': 'meanRTz'})\
#     .sort_values('meanRTz', ascending=False)
# itype_excl_desc.to_csv('itype_desc_excl.csv')
# # print(itype_excl_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# # RT Z-score plot, image type - mutually exclusive columns
# -------------------------------------------------------------------------------------------------------------------- #
# fig2, ax = plt.subplots(figsize=(8.5, 11))
# itype_excl_desc.sort_values('meanRTz').plot(kind='barh', y='meanRTz', ax=ax, edgecolor='black')
# ax.axvline(0, color='black')
# ax.set_title('Image Type (Mutually Exclusive Columns) - Mean Source-Adjusted RT Scores')
# # plt.savefig('image_type_excl_rtZ.png', dpi=300, bbox_inches='tight')
# # plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# # Basic descriptives, image branding - mutually exclusive columns
# -------------------------------------------------------------------------------------------------------------------- #
# ibrand_excl_desc = irma_rel.groupby('image.brand').agg({'image.brand': 'count', 'user.username': 'nunique',
#                                                         'retweet_count': 'median', 'rel_source_retweetZ': 'mean'})\
#     .rename(columns={'image.brand': 'count', 'user.username': 'user_count', 'retweet_count': 'medRT',
#                      'rel_source_retweetZ': 'meanRTz'})\
#     .sort_values('meanRTz', ascending=False)
# ibrand_excl_desc.to_csv('ibrand_desc_excl.csv')
# # print(ibrand_excl_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# # RT Z-score plot, image branding - mutually exclusive columns
# -------------------------------------------------------------------------------------------------------------------- #
# fig3, ax = plt.subplots()
# ibrand_excl_desc.sort_values('meanRTz').plot(kind='barh', y='meanRTz', ax=ax, edgecolor='black')
# ax.axvline(0, color='black')
# ax.set_title('Image Branding (Mutually Exclusive Columns) - Mean Source-Adjusted RT Scores')
# # plt.savefig('image_brand_excl_rtZ.png', dpi=300, bbox_inches='tight')
# # plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Basic descriptives, image type - joined columns
# -------------------------------------------------------------------------------------------------------------------- #
# itype_join_desc = irma_rel.groupby('image_join').agg({'image_join': 'count', 'user.username': 'nunique',
#                                                       'retweet_count': 'median', 'rel_source_retweetZ': 'mean'})\
#     .rename(columns={'image_join': 'count', 'user.username': 'user_count', 'retweet_count': 'medRT',
#                      'rel_source_retweetZ': 'meanRTz'})\
#     .sort_values('meanRTz', ascending=False)
# itype_join_desc.to_csv('itype_desc_join.csv')
# itype_join_desc = itype_join_desc.loc[itype_join_desc['count'] >= 30]
# #print(itype_join_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# RT Z-score plot, image type - joined columns
# -------------------------------------------------------------------------------------------------------------------- #
# fig4, ax = plt.subplots(figsize=(8.5, 11))
# itype_join_desc.sort_values('meanRTz').plot(kind='barh', y='meanRTz', ax=ax, edgecolor='black')
# ax.axvline(0, color='black')
# ax.set_title('Image Type (Joined Columns; Count >= 30) - Mean Source-Adjusted RT Scores')
# # plt.savefig('image_type_join_rtZ.png', dpi=300, bbox_inches='tight')
# # plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# # Basic descriptives, image branding - joined columns
# -------------------------------------------------------------------------------------------------------------------- #
# ibrand_join_desc = irma_rel.groupby('brand_join').agg({'brand_join': 'count', 'user.username': 'nunique',
#                                                        'retweet_count': 'median', 'rel_source_retweetZ': 'mean'})\
#     .rename(columns={'brand_join': 'count', 'user.username': 'user_count', 'retweet_count': 'medRT',
#                      'rel_source_retweetZ': 'meanRTz'})\
#     .sort_values('meanRTz', ascending=False)
# ibrand_join_desc.to_csv('ibrand_desc_join.csv')
# ibrand_join_desc = ibrand_join_desc.loc[ibrand_join_desc['count'] >= 30]
# # print(ibrand_join_desc)

# -------------------------------------------------------------------------------------------------------------------- #
# # RT Z-score plot, image branding - joined columns
# -------------------------------------------------------------------------------------------------------------------- #
# fig5, ax = plt.subplots()
# ibrand_join_desc.sort_values('meanRTz').plot(kind='barh', y='meanRTz', ax=ax, edgecolor='black')
# ax.axvline(0, color='black')
# ax.set_title('Image Branding (Joined Columns; Count >= 30) - Mean Source-Adjusted RT Scores')
# # plt.savefig('image_brand_join_rtZ.png', dpi=300, bbox_inches='tight')
# # plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Image type bubble plot timeline (with original image codes)
# -------------------------------------------------------------------------------------------------------------------- #
# ttk.timeline(df=irma_rel,
#              value_cols=list(image_dict.keys()),
#              values=[1],
#              labels=list(image_dict.values()),
#              size_col='retweet_count',
#              color_col='image.brand',
#              color_vals=['nws', 'non_nws', 'multiple', 'no_branding'],
#              clabel_dict=brand_dict,
#              palette=brand_palette,
#              ctitle='brand',
#              dates=irma_dates,
#              datetime_col='created_at',
#              show=True,
#              save=False)

# -------------------------------------------------------------------------------------------------------------------- #
# Image brand bubble plot timeline
# -------------------------------------------------------------------------------------------------------------------- #
# ttk.timeline(df=irma_rel,
#              value_cols=list(brand_dict.keys()),
#              values=[1],
#              labels=list(brand_dict.values()),
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              color_vals=['No', 'Yes'],
#              dates=irma_dates,
#              datetime_col='created_at',
#              show=False,
#              save=False)

# -------------------------------------------------------------------------------------------------------------------- #
# # Image type, joined columns, bubble plot timeline
# -------------------------------------------------------------------------------------------------------------------- #
# image_joined30 = itype_join_desc.loc[itype_join_desc['count'] >= 30].index.drop_duplicates().tolist()
# print(image_joined30)
# irma_rel_image_joined30 = irma_rel.loc[irma_rel['image_join'].isin(image_joined30)]
# ttk.timeline(df=irma_rel,
#              value_cols=['image_join'],
#              values=irma_rel_image_joined30['image_join'].drop_duplicates().tolist(),
#              labels=irma_rel_image_joined30['image_join'].drop_duplicates().tolist(),
#              size_col='retweet_count',
#              color_col='image.brand',
#              color_vals=['nws', 'non_nws', 'multiple', 'no_branding'],
#              dates=irma_dates,
#              datetime_col='created_at',
#              show=True,
#              save=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Visualize overlaps for one image type at a time
# -------------------------------------------------------------------------------------------------------------------- #
# irma_rel_cone = irma_rel.loc[irma_rel['image_type.ww'] == 1]
# print(len(irma_rel_cone))
#
# # cone_overlap_cols = []
# # for col in list(image_dict.keys()):
# #     if irma_rel_cone[col].sum() > 0:
# #         cone_overlap_cols.append(col)
# #
# # cone_overlap_labels = [image_dict[i] for i in cone_overlap_cols]
# #
# # print(cone_overlap_labels)
#
# cone_join = irma_rel_cone.groupby('image_join').agg({'image_join': 'count'}).rename(columns={'image_join': 'count'})\
#     .sort_values('count', ascending=False)
# print(cone_join)
# cone_join_labels = cone_join.loc[cone_join['count'] >= 10].index.tolist()
#
# ttk.timeline(df=irma_rel_cone,
#              value_cols=['image_join'],
#              values=cone_join_labels,
#              labels=cone_join_labels,
#              size_col='retweet_count',
#              color_col='bot_tweet',
#              color_vals=['No', 'Yes'],
#              dates=irma_dates,
#              datetime_col='created_at',
#              show=True,
#              save=False)
# </editor-fold>

# <editor-fold desc="Comparing tweet counts for coded categories based on different time filters"
# Comparing tweet counts for coded categories based on different time filters on Sep 9
irma_tf1 = irma_rel.loc[irma_rel['created_at'] < tz.localize(datetime(2017, 9, 9, 0))]
irma_tf2 = irma_rel.loc[irma_rel['created_at'] < tz.localize(datetime(2017, 9, 9, 12))]
irma_tf3 = irma_rel.loc[irma_rel['created_at'] < tz.localize(datetime(2017, 9, 9, 15))]
irma_tf4 = irma_rel.loc[irma_rel['created_at'] < tz.localize(datetime(2017, 9, 9, 18))]
irma_tf5 = irma_rel.loc[irma_rel['created_at'] < tz.localize(datetime(2017, 9, 10, 0))]
irma_tfs = [irma_rel, irma_tf1, irma_tf2, irma_tf3, irma_tf4, irma_tf5]
times = ['No Filter', 'Sep 9 - 12 AM', 'Sep 9 - 12 PM', 'Sep 9 - 3 PM', 'Sep 9 - 6 PM', 'Sep 10 - 12 AM']
df_sum = pd.DataFrame()
names = []
for i, irma_tf in enumerate(irma_tfs):
    counts, names, cut_time = ([] for j in range(3))
    names.append('overall')
    counts.append(len(irma_tf))

    for haz in list(hazard_dict.keys()):
        names.append(haz)
        counts.append(len(irma_tf.loc[irma_tf['hazard'] == haz]))

    for risk in list(risk_dict.keys()):
        names.append(risk)
        counts.append(len(irma_tf.loc[irma_tf['risk'] == risk]))

    for img_col in list(image_dict.keys()):
        names.append(image_dict[img_col])
        counts.append(len(irma_tf.loc[irma_tf[img_col] == 1]))

    df_sum[times[i]] = counts

df_sum['code'] = names
df_sum = df_sum.set_index('code')
print(df_sum)
df_sum.to_csv('irma_tf_comp.csv')
# </editor-fold>
