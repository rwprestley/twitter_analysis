import numpy as np
import pandas as pd
from tweet_coding.tweet_coding import risk_image_coding, rel_fore_coding, image_coding
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
f = 'new_missing.csv'
fr = 'new_missing_risk.csv'
frf = 'new_missing_rel_fore.csv'
fimg = 'new_missing_image.csv'
d = 'New Missing (Dec 2020)'

risk_image_coding(fr, f, d, "Risk image", ['hrisk_img'])
rel_fore_coding(frf, fr, d, "Relevance and forecast", ['relevant', 'forecast'])
image_coding(fimg, frf, d, "Image")
