import pandas as pd
import twitter_toolkit as ttk
from datetime import datetime
import pytz

pd.options.mode.chained_assignment = None

# <editor-fold desc="Merge data and progressively filter/clean Harvey dataset">
tweets_harvey = ttk.merge(json_file='Data\\harvey_tweets.json', himn_file='Data\\harvey_irma_twitter_data.csv',
                          missing_file='Data\\tweet_data_missing_forecast.csv')

col_order_df = pd.read_csv('col_order.csv')
col_order = col_order_df['New Order'].tolist()
tweets_harvey_calc = ttk.tweet_diffusion_calc(tweet_df=tweets_harvey, diff_folder='Data\\harvey_tweet_diffusion_files',
                                              col_order=col_order, tweet_df_name='Data\\tweets_harvey_calc')

tweets_harvey_final = ttk.image_filter(tweets_harvey_calc)
tweets_harvey_final = ttk.scope_aff_filter(tweets_harvey_final, col_order=col_order, sep_exp=True)
tweets_harvey_final = ttk.tweet_filter(tweets_harvey_final, rt_range=range(0, 10000))

# Convert the tweet created at column to datetime format.
tweets_harvey_final['tweet-created_at'] = pd.to_datetime(tweets_harvey_final['tweet-created_at'],
                                                         format='%Y-%m-%d %H:%M:%S%z')
# </editor-fold>

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
images_tl = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
             'Rainfall Forecast/Outlook', 'River Flood Forecast', 'SPC Convective Products', 'Cone',
             'Watch/Warning (Exp)', 'Watch/Warning']
images_tl_no_ww = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
                   'Rainfall Forecast/Outlook', 'River Flood Forecast', 'SPC Convective Products', 'Cone']
images_tl_no_conv = ['Multiple', 'Other - Non-Forecast', 'Other - Forecast', 'Key Messages', 'Model Output',
                     'Rainfall Forecast/Outlook', 'River Flood Forecast', 'Cone']
images = images_tl[::-1]
images_no_ww = images_tl_no_ww[::-1]
images_no_conv = images_tl_no_conv[::-1]

images_ww_map = ['Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'Non-WW', 'WW (Exp)',
                 'WW (Non-Exp)']
images_ww_dict = dict(zip(images_tl, images_ww_map))
# </editor-fold>

# <editor-fold desc="Sources">
sources = ['National NWS (Exp)', 'National NWS', 'National Wx Media', 'National Other', 'Local NWS (Exp)', 'Local NWS',
           'Local Wx Media', 'Local News Media', 'Local Wx Bloggers', 'Local Other']
sources_no_ww = ['National NWS', 'National Wx Media', 'National Other', 'Local NWS', 'Local Wx Media',
                 'Local News Media', 'Local Wx Bloggers', 'Local Other']
sources_local = ['Local NWS', 'Local Wx Media', 'Local News Media', 'Local Wx Bloggers', 'Local Other']
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

