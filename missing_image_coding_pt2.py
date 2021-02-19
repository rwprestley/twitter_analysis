import numpy as np
import pandas as pd
from tweet_coding.tweet_coding import risk_image_coding, rel_fore_coding, image_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'new_missing_new.csv'
fr = 'new_missing_risk_new.csv'
frf = 'new_missing_rel_fore_new.csv'
fimg = 'new_missing_image_new.csv'
d = 'New Missing (Dec 2020)'
datecol = 'created_at'

risk_image_coding(fr, f, d, "Risk image", ['hrisk_img'], datecol)
rel_fore_coding(frf, fr, d, "Relevance and forecast", ['relevant', 'forecast'], datecol)
image_coding(fimg, frf, d, "Image", datecol)