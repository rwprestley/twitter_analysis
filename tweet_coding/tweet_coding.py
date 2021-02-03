import pandas as pd
from .coding import *

all_in = ['y', 'Yes', 'Y', 'yes', '1', 'n', 'No', 'N', 'no', '0', 'na', 'nan', 'none', '', 'Not Available']
all_out = ['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'Not Available', 'Not Available',
           'Not Available', '', 'Not Available']


def risk_image_coding(cfile, dfile, direc, ctype, codecols):
    """
    Code tweets for hurricane risk imagery

    Parameters:
        cfile:
        dfile:
        direc:
        ctype:
        codecols:
    """
    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, codecols)

    # For each uncoded tweet...
    for i in range(0, len(to_code)):
        # Display each tweet in an incognito Chrome tab
        display_tweet(to_code['link'].iloc[i])

        # Code the uncoded tweets for presence of risk images
        risk_in = code_tweet('Does this tweet contain a hurricane risk image?', all_in)

        # Add coded input value to dataframe
        to_code['hrisk_img'].iloc[i] = risk_in

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, codecols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc)

        # Display coding progress.
        display_progress(tweetdata, codecols, ctype, ['yes', 'no', 'Not Available'])


def rel_fore_coding(cfile, dfile, direc, ctype, codecols):
    """
    Code tweets for relevant, forecast information

    Parameters:
        cfile:
        dfile:
        direc:
        ctype:
        codecols:
    """
    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, codecols, select_col='hrisk_img',
                                            select_crit='yes')

    # For each tweet yet to be coded...
    for i in range(0, len(to_code)):
        # Display each tweet in an incognito Chrome tab
        display_tweet(to_code['link'].iloc[i])

        # Code the tweets for relevance and forecast information
        rel_in = code_tweet('Is this tweet relevant to Hurricane Harvey?', all_in)
        fore_in = code_tweet('Does this tweet contain forecast information relevant to Harvey?', all_in)

        # Add coded input value to dataframe
        to_code['relevant'].iloc[i] = rel_in
        to_code['forecast'].iloc[i] = fore_in

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, codecols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc)

        # Display coding progress.
        display_progress(tweetdata, codecols, ctype, ['yes', 'no', 'Not Available'])


def image_coding(cfile, dfile, direc, ctype):
    """
     Code tweets based on their image content

     Parameters:
         cfile:
         dfile:
         direc:
         ctype:
     """
    # Define column names for each image code.
    image_cols = ['trop-out', 'cone', 'arrival', 'prob', 'surge', 'key-msg', 'ww', 'threat-impact', 'conv-out',
                  'meso-disc', 'rain-fore', 'rain-out', 'riv-flood', 'spag', 'text-img', 'model', 'evac',
                  'other-fore', 'other-non-fore', 'video']
    ww_cols = ['ww_exp', 'ww_cone', 'ww_md']
    other_cols = ['official', 'unofficial', 'spanish']
    all_cols = image_cols + ww_cols + other_cols

    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, all_cols, select_col='forecast',
                                            select_crit='yes')

    # For each tweet yet to be coded...
    for i in range(0, len(to_code)):

        # Display the tweet in an incognito Chrome tab
        display_tweet(to_code['link'].iloc[i])

        # Tell user if tweet media is video
        if to_code['media.media_type'].iloc[i] == 'video':
            print('This is a video')

        # Code the tweets on image content
        nan_in = ['0', 'na', 'nan', 'none', '', 'Not Available']
        all_img_in = image_cols + nan_in
        img_in = code_tweet('Which image types are present in this tweet?', all_img_in)

        # If watch/warning is coded, ask users follow-up questions about the type of watch/warning
        ww_exp_in, ww_cone_in, ww_md_in = code_ww_tweet(img_in, all_in, ['n', 'No', 'N', 'no', '0'])

        # Obtain user input for image branding and language codes.
        off_in = code_tweet('Does this tweet contain an officially branded image?', all_in)
        unoff_in = code_tweet('Does this tweet contain an unofficially branded image?', all_in)
        lang_in = code_tweet('Does this image contain information provided in Spanish?', all_in)

        # Add coded input value to dataframe.
        for col in image_cols:
            if col in img_in:
                to_code[col].iloc[i] = ['y']
            else:
                to_code[col].iloc[i] = ['n']

        to_code['ww_exp'].iloc[i] = ww_exp_in
        to_code['ww_cone'].iloc[i] = ww_cone_in
        to_code['ww_md'].iloc[i] = ww_md_in
        to_code['official'].iloc[i] = off_in
        to_code['unofficial'].iloc[i] = unoff_in
        to_code['spanish'].iloc[i] = lang_in

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, all_cols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc)

        # Display coding progress.
        display_progress(tweetdata, all_cols, ctype, ['yes'])
