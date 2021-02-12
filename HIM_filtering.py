import pandas as pd
import numpy as np
import json
import gzip
from datetime import datetime
import pytz

# Read in original (uncoded) HIMN data
himn_16k = pd.read_csv('HIMN_official_forecast_tweet_data_updated_qt (1).csv', low_memory=False,
                       header=0, encoding="ISO-8859-1")
himn_16k['tweet-id'] = himn_16k['tweet-id'].astype(np.int64)

# Filter original HIMN data by time
timezone = pytz.timezone('UTC')
harvey_start = timezone.localize(datetime(2017, 8, 17, 0))
harvey_end = timezone.localize(datetime(2017, 9, 2, 15))

himn_16k['tweet-created_at'] = pd.to_datetime(himn_16k['tweet-created_at'])
himn_16k['tweet-created_at'] = himn_16k['tweet-created_at'].dt.tz_localize('UTC')
himn_16k_ht = himn_16k.loc[(himn_16k['tweet-created_at'] >= harvey_start) &
                           (himn_16k['tweet-created_at'] <= harvey_end)]

# Read in original, coded HIMN data.
himn_16k_coded = pd.read_csv('tweets_all.csv', low_memory=False, header=0, encoding="ISO-8859-1")
himn_16k_rel = himn_16k_coded.loc[himn_16k_coded['rel_Harvey'] == '1']
himn_16k_fore = himn_16k_rel.loc[himn_16k_rel['fore_Harvey'] == '1']

# Read in originator codes from original 16k risk-image data.
origs_16k_db = pd.read_csv('final_originator_codes.csv')
origs_16k_db['Originator'] = origs_16k_db['Originator'].str.lower()
origs_16k = origs_16k_db['Originator'].tolist()

# Read in tweet-ids from original all-media data.
himn_85k = pd.read_csv('orig_85k_media_tweet_ids.txt', header=None)
himn_85k.columns = ['tweet-id']

# Read simplified tweet data JSON.
sd = [json.loads(line) for line in gzip.open(r"tweets_116k_simple.json.gz", 'rb')]
him_116k = pd.json_normalize(sd)
him_116k['text'] = him_116k['text'].str.encode('UTF-8').str.decode('UTF-8')
him_116k['user.name'] = him_116k['user.name'].str.encode('UTF-8').str.decode('UTF-8')
him_116k['text'] = him_116k['text'].str.replace('&amp;', '&')
him_116k['id.trunc'] = him_116k['id'].str[:15]
him_116k['id'] = him_116k['id'].astype(np.int64)
him_116k['created_at'] = pd.to_datetime(him_116k['created_at'])
him_116k['user.screen_name'] = him_116k['user.screen_name'].str.lower()
him_116k.sort_values('created_at', inplace=True)

# Filter to only include tweets posted during the Harvey timeframe.
him_ht = him_116k.loc[(him_116k['created_at'] >= harvey_start) & (him_116k['created_at'] <= harvey_end)]
print('Harvey time-filtered: ' + str(len(him_ht)))

# Count the number of originators in the Harvey time-filtered dataset
origs_him_ht = him_ht['user.screen_name'].drop_duplicates().tolist()
print('Time-filtered Originators: ' + str(len(origs_him_ht)))

# Create the union of the original originator list and new missing list.
origs_all = list(set(origs_16k) | set(origs_him_ht))
print('HIMN Union Originators: ' + str(len(origs_all)))

# Compare originators in Harvey time-filtered dataset to original 16k risk image dataset
origs_new_ht = [orig for orig in origs_him_ht if orig not in origs_16k]
print('New originators: ' + str(len(origs_new_ht)))

# Read in new coded originators
origs_new_ht_coded = pd.read_csv('origs_missing_coded.csv')

# Filter to only include tweets from local users and national users with NWS or weather media affiliations (after coding
# new/uncoded originators).
origs_sf = origs_new_ht_coded.loc[(origs_new_ht_coded['Scope'] == 'Local - Harvey') |
                                  ((origs_new_ht_coded['Scope'] == 'National/International') &
                                   ((origs_new_ht_coded['Affiliation'] == 'Gov - Wx - NWS') |
                                    (origs_new_ht_coded['Affiliation'] == 'Media - Wx')))]['Originator'].\
                                                drop_duplicates().tolist()
him_sf = him_ht.loc[him_ht['user.screen_name'].isin(origs_sf)]
print('Time and source filtered: ' + str(len(him_sf)))

# Calculate the number of unique authoritative sources in the Harvey time and source filtered dataset
origs_him_sf = him_sf['user.screen_name'].drop_duplicates().tolist()
print('Time and source-filtered originators: ' + str(len(origs_him_sf)))

# Filter to only include tweets that weren't already included in the 85k database or the 16k database.
himn_new = him_sf.loc[(~him_sf['id'].isin(himn_85k['tweet-id'].tolist())) &
                      (~him_sf['id'].isin(himn_16k['tweet-id'].tolist()))]

# Count the number of tweets in the Harvey time and source filtered dataset that were previously coded as a risk image
# by Bica et al
old_risk = him_sf.loc[him_sf['id'].isin(himn_16k['tweet-id'].tolist())]
to_code = him_sf.loc[(~him_sf['id'].isin(himn_16k['tweet-id'].tolist())) &
                     (~him_sf['id'].isin(himn_85k['tweet-id'].tolist()))]

# Count the number of new tweets coded as a risk image
risk_coded = pd.read_csv('New Missing (Dec 2020)\\new_missing_risk.csv')
new_risk = risk_coded.loc[risk_coded['hrisk_img'] == 'yes']

# Combine the old and new risk datasets
all_risk = pd.concat([old_risk, new_risk])
print('Risk image filtering: ' + str(len(all_risk)))

# Count the number of unique originators in the combined risk image dataset
origs_risk = all_risk['user.screen_name'].drop_duplicates().tolist()
print('Risk filtering originators: ' + str(len(origs_risk)))

# Count the number of tweets coded as risk images in the Harvey time and source filtered dataset that were coded as
# relevant/containing forecast information
himn_old_coded = pd.read_csv('Data\\harvey_irma_twitter_data.csv', encoding="ISO-8859-1")
himn_old_coded['id.trunc'] = himn_old_coded['tweet-id'].astype(str).str[:15]
old_risk = old_risk.merge(himn_old_coded[['id.trunc', 'rel_Harvey', 'fore_Harvey']], how='left', on='id.trunc')
old_rel = old_risk.loc[old_risk['rel_Harvey'] == "1"]
old_fore = old_risk.loc[old_risk['fore_Harvey'] == "1"]

# Count the number of new tweets coded as relevant/containing forecast information
rel_fore_coded = pd.read_csv('New Missing (Dec 2020)\\new_missing_rel_fore.csv')
new_rel = rel_fore_coded.loc[rel_fore_coded['relevant'] == 'yes']
new_fore = rel_fore_coded.loc[rel_fore_coded['forecast'] == 'yes']

# Combine old and new relevant/forecast datasets
all_rel = pd.concat([old_rel, new_rel])
all_fore = pd.concat([old_fore, new_fore])
print('Relevant tweets: ' + str(len(all_rel)))
print('Forecast tweets: ' + str(len(all_fore)))

# Count the number of unique originators in the combined relevant/forecast datasets
origs_rel = all_rel['user.screen_name'].drop_duplicates().tolist()
origs_fore = all_fore['user.screen_name'].drop_duplicates().tolist()
print('Relevant originators: ' + str(len(origs_rel)))
print('Forecast originators: ' + str(len(origs_fore)))

# Initialize a database to store counts for various filtered databases for each originator in the union of the original
# originator list and the new missing list
origs_all_count_in = pd.DataFrame()
origs_all_count_in['Originator'] = origs_all
origs_all_count_in = origs_all_count_in.merge(origs_new_ht_coded[['Originator', 'Scope', 'Affiliation', 'Agency']],
                                              how='outer',
                                              on='Originator')
origs_all_count_in.drop_duplicates(subset=['Originator'], inplace=True)

# Count counts and status in databases for each originator
orig_in_16k, orig_in_16k_rel, orig_in_16k_fore, orig_in_missing, orig_in_new_risk, orig_in_new_rel, orig_in_new_fore, \
    count_16k, count_16k_rel, count_16k_fore, count_missing, count_new_risk, count_new_rel, count_new_fore = \
    ([] for i in range(14))

for orig in origs_all:
    count_16k.append(len(himn_16k.loc[himn_16k['tweet-user_screen_name'] == orig]))
    count_16k_rel.append(len(himn_16k_rel.loc[himn_16k_rel['tweet-user_screen_name'] == orig]))
    count_16k_fore.append(len(himn_16k_fore.loc[himn_16k_fore['tweet-user_screen_name'] == orig]))
    count_missing.append(len(himn_new.loc[himn_new['user.screen_name'] == orig]))
    count_new_risk.append(len(new_risk.loc[new_risk['user.screen_name'] == orig]))
    count_new_rel.append(len(new_rel.loc[new_rel['user.screen_name'] == orig]))
    count_new_fore.append(len(new_fore.loc[new_fore['user.screen_name'] == orig]))

    orig_in_16k.append(['Yes' if orig in origs_16k else 'No'][0])
    orig_in_16k_rel.append(['Yes' if orig in himn_16k_rel['tweet-user_screen_name'].drop_duplicates().tolist() else
                            'No'][0])
    orig_in_16k_fore.append(['Yes' if orig in himn_16k_fore['tweet-user_screen_name'].drop_duplicates().tolist() else
                             'No'][0])
    orig_in_missing.append(['Yes' if orig in origs_new_ht else 'No'][0])
    orig_in_new_risk.append(['Yes' if orig in new_risk['user.screen_name'].drop_duplicates().tolist() else 'No'][0])
    orig_in_new_rel.append(['Yes' if orig in new_rel['user.screen_name'].drop_duplicates().tolist() else 'No'][0])
    orig_in_new_fore.append(['Yes' if orig in new_fore['user.screen_name'].drop_duplicates().tolist() else 'No'][0])

origs_all_count_in['Original (16K) count'] = count_16k
origs_all_count_in['Original (16K) relevant'] = count_16k_rel
origs_all_count_in['Original (16K) forecast'] = count_16k_fore
origs_all_count_in['Missing count'] = count_missing
origs_all_count_in['Missing risk count'] = count_new_risk
origs_all_count_in['Missing relevant'] = count_new_rel
origs_all_count_in['Missing forecast count'] = count_new_fore
origs_all_count_in['in_16k'] = orig_in_16k
origs_all_count_in['in_16k_rel'] = orig_in_16k_rel
origs_all_count_in['in_16k_fore'] = orig_in_16k_fore
origs_all_count_in['in_missing'] = orig_in_missing
origs_all_count_in['in_missing_risk'] = orig_in_new_risk
origs_all_count_in['in_missing_rel'] = orig_in_new_rel
origs_all_count_in['in_missing_fore'] = orig_in_new_fore
origs_all_count_in.to_csv('origs_all_count_in.csv')
