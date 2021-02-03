import webbrowser


def display_tweet(url):
    """
    Display a tweet in a Chrome incognito tab

    Parameters:
        url: A text string denoting the URL of a tweet
    """

    # Display the tweet in an incognito Chrome tab.
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s --incognito'
    webbrowser.get(chrome_path).open_new_tab(url)