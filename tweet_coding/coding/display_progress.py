def display_progress(tweetdata, codecols, ctype, disptypes):
    """
    Print coding progress, overall and for specified coded columns

    Parameters:
        tweetdata: A Pandas dataframe of all tweets (coded or not)
        codecols: A list of strings, where the strings denote the coded columns that progress should be displayed for
        ctype: A text string denoting the type of coding being done (e.g. 'Risk image', 'Relevance and forecast'). First
                  letter should be capitalized.
        disptypes: A list of strings, where the strings denote the coded categories that progress should be
                       displayed for
    """
    # Display overall coding progress.
    coded = len(tweetdata.loc[tweetdata[codecols[0]] != ''])
    tot = len(tweetdata)
    per_comp = '{:.1%}'.format(coded / tot)
    print(ctype + ' coding completed: ' + str(coded) + '/' + str(tot) + ' (' + str(per_comp) + ')')

    # Display coding results for each provided data column.
    for col in codecols:
        print_str = col + ':'
        for disp in disptypes:
            val = len(tweetdata.loc[tweetdata[col] == disp])
            val_per = '{:.1%}'.format(val / coded)
            print_str = print_str + ' ' + str(val) + ' ' + disp + ' (' + str(val_per) + '),'

        print_str = print_str[:-1]
        print(print_str)
