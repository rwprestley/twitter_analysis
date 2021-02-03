def map_input(to_code, codecols, in_vals, out_vals):
    """
        Map user input values to "yes", "no", or "Not Available"

        Parameters:
            to_code: A Pandas dataframe of tweets to be coded
            codecols: A list of strings, where each string represents a column where coded data is stored
            in_vals: A list of strings, where each string is an acceptable user input value
            out_vals: A list of strings, which maps acceptable user input to "yes", "no", or "Not Available"
        """
    for col in codecols:
        to_code[col] = [''.join(map(str, l)) for l in to_code[col]]
        to_code[col] = to_code[col].map(dict(zip(in_vals, out_vals)))
    return to_code