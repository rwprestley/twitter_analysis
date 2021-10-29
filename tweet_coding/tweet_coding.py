import pandas as pd
import numpy as np
from .coding import *

yes_vals = ['y', 'Yes', 'Y', 'yes', '1']
no_vals = ['n', 'No', 'N', 'no', '0']
all_in = ['y', 'Yes', 'Y', 'yes', '1', 'n', 'No', 'N', 'no', '0', 'na', 'nan', 'none', '', 'Not Available']
all_out = ['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'Not Available', 'Not Available',
           'Not Available', '', 'Not Available']


def risk_image_coding(cfile, dfile, direc, ctype, codecols, datecol):
    """
    Code tweets for hurricane risk imagery

    Parameters:
        cfile:
        dfile:
        direc:
        ctype:
        codecols:
        datecol: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
    """
    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, codecols, datecol)

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
        tweetdata = save_coding(coded, to_code, cfile, direc, datecol)

        # Display coding progress.
        display_progress(tweetdata, codecols, ctype, ['yes', 'no', 'Not Available'])


def filter_coding(cfile, dfile, direc, ctype, codecols, datecols):
    """
    Code tweets for relevant, forecast information

    Parameters:
        cfile:
        dfile:
        direc:
        ctype:
        codecols:
        datecols: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
    """
    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, codecols, datecols)

    # For each tweet yet to be coded...
    for i in range(0, len(to_code)):
        # Display each tweet in an incognito Chrome tab
        display_tweet(to_code['tweet-url'].iloc[i])

        # Code the tweets for relevance and forecast information
        del_qt_in = code_tweet('\nIs this a deleted quote tweet?', all_in)

        if set(del_qt_in).issubset(set(no_vals)) is True:
            rel_in = code_tweet('\nIs this tweet relevant to Hurricane Irma?', all_in)

            if set(rel_in).issubset(set(yes_vals)) is True:
                spanish_in = code_tweet('\nDoes this tweet contain information relevant to Hurricane Irma in Spanish?',
                                        all_in)

                #if set(spanish_in).issubset(set(yes_vals)) is True:
                    #can_code_in = code_tweet('Can a coding judgement be made for forecast and local relevance?', all_in)

                    #if set(can_code_in).issubset(set(yes_vals)) is True:
                        #fore_in = code_tweet('Does this tweet contain forecast information relevant to Irma?', all_in)
                        #loc_rel_in = code_tweet('Does this tweet contain Irma information relevant to areas of the '
                        #                        'continental United States affected by Irma?', all_in)

                    #else:
                        #fore_in = ''
                        #loc_rel_in = ''

                #else:
                    #fore_in = code_tweet('Does this tweet contain forecast information relevant to Irma?', all_in)
                    #loc_rel_in = code_tweet('Does this tweet contain Irma information relevant to areas of the '
                     #                       'continental United States affected by Irma?', all_in)

            else:
                spanish_in = ''
                #fore_in = ''
                #loc_rel_in = ''

        else:
            rel_in = ''
            spanish_in = ''
            #fore_in = ''
            #loc_rel_in = ''

        # Add coded input value to dataframe
        to_code['deleted_qt'].iloc[i] = del_qt_in
        to_code['relevant'].iloc[i] = rel_in
        to_code['spanish'].iloc[i] = spanish_in
        #to_code['forecast'].iloc[i] = fore_in
        #to_code['local_relevant'].iloc[i] = loc_rel_in

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, codecols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc, datecols)

        # Display coding progress.
        display_progress(tweetdata, codecols, ctype, ['yes', 'no', 'Not Available'])


def image_coding(cfile, dfile, direc, ctype, datecol):
    """
     Code tweets based on their image content

     Parameters:
         cfile:
         dfile:
         direc:
         ctype:
         datecol: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
     """
    # Define column names for each image code.
    image_cols = ['trop-out', 'cone', 'arrival', 'prob', 'surge', 'key-msg', 'ww', 'threat-impact', 'conv-out',
                  'meso-disc', 'rain-fore', 'rain-out', 'riv-flood', 'spag', 'text-img', 'model', 'evac',
                  'other-fore', 'other-non-fore', 'video']
    ww_cols = ['ww_exp', 'ww_cone', 'ww_md']
    other_cols = ['official', 'unofficial', 'spanish']
    all_cols = image_cols + ww_cols + other_cols

    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, all_cols, datecol, select_col='forecast',
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
        tweetdata = save_coding(coded, to_code, cfile, direc, datecol)

        # Display coding progress.
        display_progress(tweetdata, all_cols, ctype, ['yes'])


def md_coding(cfile, dfile, direc, ctype, codecols, datecol):
    """
       Code mesoscale discussion tweets into SPC and WPC versions

       Parameters:
           cfile: A text string denoting what the coded data should be saved as
           dfile: A text string denoting the existing data that should be coded
           direc: A text string denoting the directory that the existing data and newly coded data should be saved in
           ctype: A text string denoting the type of coding being done (e.g. 'Risk image', 'Relevance and forecast').
                      First letter should be capitalized.
           codecols: A list of strings, where each string represents a column where coded data is stored
           datecol: A text string denoting the Pandas column name of the data to be saved that represents the date and
                         time
       """
    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, codecols, datecol, select_col='image-type_meso-disc',
                              select_crit=1)

    # For each tweet yet to be coded...
    for i in range(0, len(to_code)):
        # Display each tweet in an incognito Chrome tab
        display_tweet(to_code['tweet-url'].iloc[i])

        # Code the tweets for relevance and forecast information
        spc_in = code_tweet('Is this mesoscale discussion provided by the SPC?', all_in)
        wpc_in = code_tweet('Is this mesoscale discussion provided by the WPC?', all_in)

        # Add coded input value to dataframe
        to_code['image-type_meso-disc_spc'].iloc[i] = spc_in
        to_code['image-type_meso-disc_wpc'].iloc[i] = wpc_in

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, codecols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc, datecol)

        # Display coding progress.
        display_progress(tweetdata, codecols, ctype, ['yes'])


def hazard_risk_coding(cfile, dfile, direc, ctype, datecols, url_col):
    """
        Code tweets for hazard and risk information type

        Parameters:
            cfile: A text string denoting what the coded data should be saved as
            dfile: A text string denoting the existing data that should be coded
            direc: A text string denoting the directory that the existing data and newly coded data should be saved in
            ctype: A text string denoting the type of coding being done (e.g. 'Risk image', 'Relevance and forecast').
                       First letter should be capitalized.
            datecols: A list of column(s) to sort the data by (e.g. date, username, etc.)
            url_col: A text string denoting the column name of the tweet URL field
        """

    # Define column names for each hazard and risk code.
    hazard_cols = ['tc', 'surge', 'rain/flood', 'convective', 'haz_other', 'haz_mult']
    risk_cols = ['forecast', 'ww', 'obs', 'past', 'risk_other', 'risk_mult']
    all_cols = hazard_cols + risk_cols

    # Read and parse Twitter data
    coded, to_code = readdata(cfile, dfile, direc, ctype, all_cols, datecols)

    # For each tweet yet to be coded...
    for i in range(0, len(to_code)):

        # Display the tweet in an incognito Chrome tab
        display_tweet(to_code[url_col].iloc[i])

        # Code the tweets for hazard
        nan_in = ['0', 'na', 'nan', 'none', '', 'Not Available']
        haz_in_cols = [col for col in hazard_cols if col != 'haz_mult']
        all_haz_in = haz_in_cols + nan_in
        haz_in = code_tweet('Which hazard(s) is/are predominantly represented in this tweet?', all_haz_in)

        # Code the tweets for risk information
        risk_in_cols = [col for col in risk_cols if col != 'risk_mult']
        all_risk_in = risk_in_cols + nan_in
        risk_in = code_tweet('Which type(s) of risk information is/are predominantly represented in this tweet?',
                             all_risk_in)

        # Add hazard codes to dataframe from input
        for col in haz_in_cols:
            if col in haz_in:
                to_code[col].iloc[i] = ['y']
            else:
                to_code[col].iloc[i] = ['n']

            # Automatically code "multiple" where multiple hazards are coded
            if len(haz_in) > 1:
                to_code['haz_mult'].iloc[i] = ['y']
            else:
                to_code['haz_mult'].iloc[i] = ['n']

        # Add risk information codes to dataframe from input
        for col in risk_in_cols:
            if col in risk_in:
                to_code[col].iloc[i] = ['y']
            else:
                to_code[col].iloc[i] = ['n']

                # Automatically code "multiple" where multiple risk information types are coded
                if len([rtype for rtype in risk_in if rtype != 'ww']) > 1:
                    to_code['risk_mult'].iloc[i] = ['y']
                else:
                    to_code['risk_mult'].iloc[i] = ['n']

        # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer
        # because they were deleted).
        to_code = map_input(to_code, all_cols, all_in, all_out)

        # Save the updated coded file after each tweet is coded.
        tweetdata = save_coding(coded, to_code, cfile, direc, datecols)

        # Display coding progress.
        display_progress(tweetdata, all_cols, ctype, ['yes'])