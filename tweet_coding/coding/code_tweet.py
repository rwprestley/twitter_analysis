import numpy as np


def code_tweet(codeq, in_vals):
    """
        Code a tweet by requesting user input to a user-provided question

        Parameters:
            codeq: A text string, where the string is the question to be asked of the user in order to code the construct
            in_vals: A list of strings, where each string is an acceptable user input value
        """

    # Obtain user input for code.
    while True:
        try:
            user_in = input(codeq)
            user_in = user_in.split(',')
        except ValueError:
            print('Oops! Try again')
            continue

        # Restrict input values to acceptable values - if user input doesn't match, prompt again.
        if (False in np.isin(user_in, in_vals)) is True:
            print('Input not acceptable. Please input one of the following: ' + str(in_vals[:-2]))
            continue
        else:
            break

    return user_in