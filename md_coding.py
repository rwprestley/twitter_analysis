import numpy as np
import pandas as pd
from tweet_coding.tweet_coding import md_coding
import os
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored
f = 'test.csv'
f_coded = 'test_md_coded.csv'
d = os.curdir
datecol = 'tweet-created_at'

# Code mesoscale-discussion tweets
md_coding(f_coded, f, d, 'Mesoscale discussion', ['image-type_meso-disc_spc', 'image-type_meso-disc_wpc'], datecol)

# Merge newly coded data with previously coded data, if not already done.
if 'image-type_meso-disc_spc' in pd.read_csv(f).columns:
    print('Mesoscale discussion coding already merged with previously coded data')

else:
    md_coded = pd.merge(pd.read_csv(f), pd.read_csv(f_coded)[
        ['tweet-id_trunc', 'image-type_meso-disc_spc', 'image-type_meso-disc_wpc']], on='tweet-id_trunc', how='outer')

    # Reformat columns
    for col in md_coded.columns:
        if col[:7] == 'Unnamed':
            md_coded.drop(col, axis=1, inplace=True)

    md_coded['image-type_meso-disc_spc'] = md_coded['image-type_meso-disc_spc'].map({'yes': 1, 'no': 0})
    md_coded['image-type_meso-disc_wpc'] = md_coded['image-type_meso-disc_wpc'].map({'yes': 1, 'no': 0})

    md_coded = md_coded.fillna(0)

    # Reorder the final dataset based on user-provided column order.
    col_order_df = pd.read_csv('col_order.csv')
    col_order = col_order_df['New Order'].tolist()
    md_coded.set_index('tweet-id', inplace=True)
    md_coded.to_csv(f)
