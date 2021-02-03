import numpy as np


def code_ww_tweet(img_in, in_vals, no_in):
    """
        Code tweets coded as containing watch/warning imagery by asking several follow-up questions about the type of
        watch/warning imagery

        Parameters:
            img_in: A list of strings, where the strings denote the image types coded for a tweet
            in_vals: A list of strings, where each string is an acceptable user input value
            no_in: A list of strings, where each string is an acceptable user input value that represents "no"
        """
    # If watch/warning is coded, ask users whether the tweet contains an experimental NWS warning image
    while ('ww' in img_in) is True:
        try:
            ww_exp_in = input('Does this tweet contain an experimental NWS warning image?')
        except ValueError:
            print('Oops! Try again')
            continue

        # Restrict input values to acceptable values - if user input doesn't match, prompt again.
        if (False in np.isin(ww_exp_in, in_vals)) is True:
            print('Input not acceptable. Please input a combination of the following: ' + str(in_vals[:-2]))
            continue

        # If the image is not an experimental NWS warning image, ask users whether the watch/warning information is
        # provided solely in conjunction with a cone image
        if (True in np.isin(ww_exp_in, no_in)) is True:
            try:
                ww_cone_in = input('Does this tweet only contain watch/warning imagery in the form of colored '
                                   'coastlines on a cone graphic?')
            except ValueError:
                print('Oops! Try again')
                continue

            # Restrict input values to acceptable values - if user input doesn't match, prompt again.
            if (False in np.isin(ww_cone_in, in_vals)) is True:
                print('Input not acceptable. Please input a combination of the following: ' + str(in_vals[:-2]))
                continue

            # If the image is not an experimental NWS warning image or solely provided in conjunction with a cone image,
            # ask users if the watch/warning information is provided solely in conjunction with a mesoscale discussion
            if (True in np.isin(ww_cone_in, no_in)) is True:
                try:
                    ww_md_in = input('Does this tweet only contain watch/warning imagery overlaid on a mesoscale '
                                     'discussion image?')
                except ValueError:
                    print('Oops! Try again')
                    continue

                # Restrict input values to acceptable values - if user input doesn't match, prompt again.
                if (False in np.isin(ww_md_in, in_vals)) is True:
                    print('Input not acceptable. Please input a combination of the following: ' + str(in_vals[:-2]))
                    continue

                else:
                    break
            else:
                ww_md_in = ['n']
                break
        else:
            ww_cone_in = ['n']
            ww_md_in = ['n']
            break
    else:
        ww_exp_in = ['n']
        ww_cone_in = ['n']
        ww_md_in = ['n']

    return ww_exp_in, ww_cone_in, ww_md_in
