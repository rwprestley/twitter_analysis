import numpy as np
import pandas as pd
import os
import webbrowser
pd.options.mode.chained_assignment = None

# Read in file and folder where coded data is stored.
file = 'new_missing_coded.csv'
direc = 'New Missing (Dec 2020)'

# If file does not exist yet, coding has not been initiated.
if (file in os.listdir(direc)) is not True:
    print('Coding has not been initiated')

    # Create a dataframe with tweets to be coded. In this case, this is all of the tweets. Initialize a column where
    # codes should be stored.
    to_code = pd.read_csv(direc + '\\new_missing.csv')
    to_code['hrisk_img'] = ''

    # Create an empty coded dataframe.
    coded = pd.DataFrame()

# If file does exist, coding has already started and should be continued where it left off.
else:
    print('Coding already started')

    # Read and clean the coded data file.
    nmiss = pd.read_csv(direc + '\\' + file)
    nmiss['hrisk_img'] = nmiss['hrisk_img'].fillna('')
    for col in nmiss.columns:
        if col[:7] == 'Unnamed':
            nmiss.drop(col, axis=1, inplace=True)

    # Split the data in to tweets that have already been coded and tweets that have yet to be coded.
    coded = nmiss.loc[nmiss['hrisk_img'] != '']
    to_code = nmiss.loc[nmiss['hrisk_img'] == '']

# For each tweet yet to be coded...
for i in range(0, len(to_code)):

    # Display the tweet in an incognito Chrome tab.
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s --incognito'
    webbrowser.get(chrome_path).open_new_tab(to_code['link'].iloc[i])

    # Obtain user input for risk image code.
    while True:
        try:
            user_in = input('Does this tweet contain a hurricane risk image?')
        except ValueError:
            print('Oops! Try again')
            continue

        # Restrict input values to acceptable values - if user input doesn't match, prompt again.
        all_in = ['y', 'Yes', 'Y', 'yes', '1', 'n', 'No', 'N', 'no', '0', 'na', 'nan', 'none', '', 'Not Available']
        if (False in np.isin(user_in, all_in)) is True:
            print('Input not acceptable. Please input one of the following: ' + str(all_in[:-2]))
            continue
        else:
            break

    # Add coded input value to dataframe.
    to_code['hrisk_img'].iloc[i] = user_in

    # Map various inputs to either 'yes', 'no', or 'Not Available' (for tweets that don't display in browswer because
    # they were deleted).
    all_out = ['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'Not Available', 'Not Available',
               'Not Available', '', 'Not Available']
    to_code['hrisk_img'] = to_code['hrisk_img'].map(dict(zip(all_in, all_out)))

    # Save the updated coded file after each tweet is coded.
    nmiss = pd.concat([coded, to_code])
    nmiss.sort_values(by='created_at', inplace=True)
    nmiss.to_csv(direc + '\\' + file)

    # Display coding progress.
    num_coded = len(nmiss.loc[nmiss['hrisk_img'] != ''])
    tot = len(nmiss)
    per_comp = '{:.2%}'.format(num_coded / tot)
    print('Risk Image Coding Completed: ' + str(num_coded) + '/' + str(tot) + ' (' + str(per_comp) + ')')
