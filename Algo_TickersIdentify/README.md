[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **TickersIdentification** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : TickersIdentification

Published in : 'Networks of News and Cross-Sectional Returns'

Description : 'Algorithm that identify the S&P500 firm tickers from news text. The algorithm only identify those tickers
in the three most commonly seen cases. Detailed description refer to the paper'

Keywords : Textual Analysis, Entity Identification

See also : ''

Author : Junjie Hu

```

### PYTHON Code
```python

"""
Python: 3.8.10
"""

import os
import pandas as pd
import datetime as dt
import re
from nltk import tokenize as token
import concurrent.futures as cf
import time
import pickle

# Symbols Extraction settings
"""
bracket_symbols_only: Find the companies only by brackets tickers if True, for both titles and contents
segment_content_match_flag: Find companies in the content using name segments if True
ticker_len_mini: Set the minimum length of the isolated tickers to find companies
single_leader_only: Choose only the articles with one leader detected if True
"""
bracket_symbols_only = False
segment_content_match_flag = True
ticker_len_mini = 3
single_leader_only = True

"""
Reg Expression Definitions:
"""
# Define the marco for regex
spx_tickers = pd.read_csv(os.getcwd() + '/Algo_TickersIdentify/SPX_TickerName.csv', index_col=0)
bracket_symbol_unrecognizable = ['TTM', 'CEO']
# 'CEO' in spx_tickers.index.tolist()
tickers_bracket = spx_tickers.index.tolist()
[tickers_bracket.remove(ele) for ele in bracket_symbol_unrecognizable if ele in tickers_bracket]

"""Create name-segments RegExp pattern"""
mapped_ticker_nameseg = pd.read_csv(os.getcwd() + '/Algo_TickersIdentify/SymbolsNSegments.csv', index_col=0)
mapped_segs = mapped_ticker_nameseg['FullSegments'].to_frame()
mapped_segs.dropna(axis=0, subset=['FullSegments'], inplace=True)
mapped_segs['ticker'] = mapped_segs.index
segments_pattern = mapped_segs.values.tolist()
seg_pattern = [[re.compile(seg_pat[0], re.IGNORECASE), seg_pat[1]] for seg_pat in segments_pattern]

"""Text cleaning regex pattern"""
remove_linebreakers_pattern = re.compile(r'[\n|\t]+')  # Remove \n and \t
sentence_break_pattern = re.compile(r'([\w\b]*?[\.?!])([A-Z]+|[A-Z][a-z]*\b)\b')  # correct sentence separation
sentence_break_connect_pattern = re.compile(r'\s[.,?!]\s')

# Remove disclaimer, clicking, and image source till the end of the sentence
disclaimer_pattern = re.compile(r'(The views and opinions expressed herein|Click here|Image source).*?(\.|\?|!|>|$)+',
                                re.IGNORECASE)
# remove click here till the end of the sentence
# remove_clicklink_pattern = re.compile(r'(Click here|Image source).*[\.?!>]+', re.IGNORECASE)
# remove copy right info till the end of the string
remove_endingtext_pattern = re.compile(r'(Copyright|All rights reserved).*', re.IGNORECASE)


def unlist(list_in):
    """
    transform possible chain list into one-dim list
    :param list_in:
    :return:
    """
    list_out = list()
    for ele in list_in:
        if type(ele) is list:
            list_out.extend(ele)
        else:
            if ele != '':
                list_out.append(ele)
            else:
                pass
    return list_out


def bracket_symbols_extract(text):
    """
    :param text: A string to be parsed
    :return: A list of symbols recognized by RE bracket_symbols_pattern
    """
    # text = content_clean_pattern.sub(repl=" ", string=text)
    # bracket_matches = bracket_symbols_pattern.findall(string=text)
    symbols_unrecog_byonlybracket = ['PEG', 'COO', 'C', 'GPS', 'AMT', 'MS', 'USB', 'FANG', 'ABC', 'MA', 'GM', 'MGM',
                                     'HD', 'CMS', 'CEO']
    lead_keys = '|'.join(
        ['NYSE', 'Nyse', 'nyse', 'NASDAQ', 'Nasdaq', 'nasdaq', 'Symbol', 'symbol', 'Symbols', 'symbols'])
    bracket_w_leadkeys_pattern = re.compile(r'\((\s*' + fr'({lead_keys})' + r'\s*:)\s*([A-Z,\s]{1,20})\s*\)')
    bracket_only_pattern = re.compile(r'\(\s*([A-Z,\s]{1,20})\s*\)')

    # Find symbols in the bracket lead by lead_keys
    bracket_leadkeys_matchs = [matched.group(3) for matched in bracket_w_leadkeys_pattern.finditer(string=text)]
    # Find the symbols in the bracket
    bracket_only_matches = [matched.group(1) for matched in bracket_only_pattern.finditer(string=text)]

    bracket_leadkeys_matchs = clean_bracket_matches(bracket_leadkeys_matchs)
    bracket_only_matches = [ele for ele in clean_bracket_matches(bracket_only_matches) if
                            ele not in symbols_unrecog_byonlybracket]

    bracket_matches = list(set(bracket_leadkeys_matchs + bracket_only_matches))

    return bracket_matches


def clean_bracket_matches(matched_list):
    matched_list = [symbol.split(',') if ',' in symbol else symbol for symbol in matched_list]
    matched_list = unlist(matched_list)
    matched_list = [ele.strip() for ele in matched_list]
    return matched_list


def unbracket_ticker_matching(text, tickers_bracket, ticker_len_mini=None):
    # Patter: (^|\s)(AAPL|AA|A)(\s|$), Symbols isolated by space or at the end/start of the string
    if ticker_len_mini is not None:
        unbracket_pattern = '|'.join([fr"{syb}" for syb in tickers_bracket if len(syb) >= ticker_len_mini])
    else:
        unbracket_pattern = '|'.join([fr"{syb}" for syb in tickers_bracket])

    unbracket_pattern = re.compile(fr'(^|\s)({unbracket_pattern})(\s|$)')
    # print(unbracket_pattern)
    # DO NOT try to find tickers that are not in the brackets within content, too dangerous
    unbracket_matched = unbracket_pattern.findall(text)
    unbracket_matched = list(set([matched[1] for matched in unbracket_matched]))
    return unbracket_matched


def symbol_extracter(headline, content, timestamp, author):
    """
    @param headline: list of lists
    @param content:  list of lists
    @return:
    """
    """
    Extract the symbols from the title and content of each news
    1. Extract symbols in brackets (title and content)
    2. Use the FullSegment to find the corresponding ticker, case-sensitive, started with white space, ended
    with white-space/./'s/
    3. Floating tickers (Full capital, titles only), bounded by white space or start/end of the string
    """

    headline_text = headline

    """Clean the text of headline and content"""
    content_text = remove_linebreakers_pattern.sub(repl=" ", string=content)  # Replace newline and tab with space
    content_text = sentence_break_pattern.sub(repl=r"\1 \2", string=content_text)  # separate two attached sentences
    content_text = sentence_break_connect_pattern.sub(r'. ', string=content_text)  # connect two sentence
    content_text = disclaimer_pattern.sub(repl='', string=content_text)  # Delete disclaimer
    # content_text = remove_clicklink_pattern.sub(repl='', string=content_text)
    content_text = remove_endingtext_pattern.sub(repl='', string=content_text)
    # print(content_text)

    # text = disclaimer_pattern.sub(repl='', string=content)
    headline_text = remove_linebreakers_pattern.sub(repl=" ",
                                                    string=headline_text)  # Replace newline and tab with space
    headline_text = sentence_break_pattern.sub(repl=" ", string=headline_text)  # Replace newline and tab with space
    headline_text = sentence_break_connect_pattern.sub(repl=" ",
                                                       string=headline_text)  # Replace newline and tab with space

    """
    Headline symbol recognition: 
    Priority: Bracket symbols > Company segment mapping > Tickers isolated in the text (for long ticker only, DANGEROUS, e.g 'A', 'T')
    """
    leaders = leaders_symbols(headline_text)
    # ==Sample TEST start:
    # headline_test_sample = ["Apple Inc. (FB) Unleashes New MacBook Pro With Touch Bar",
    #                         "Tuesday Apple Rumors: MSFT Working on iMessage Android Mockup",
    #                         "Fitbit Inc. Had A Merry Christmas, But Why Isn't Its Stock As Jolly?"]
    # correct_res = ['FB', 'AAPL', '']
    # headline_rest_res = [leaders_symbols(textitem) for textitem in headline_test_sample]
    # ==TEST END

    """Content symbol recog: """
    # Cut the content into sentences
    content_sentences = token.sent_tokenize(text=content_text, language='english')  # Split into difference sentences
    # clean_content_text = '. '.join(content_sentences)
    # sentence = content_sentences[0]
    detectedsymb_sentence = [[sentence, followers_symbols(sentence)] for sentence in content_sentences]
    detectedsymb_sentence = [item for item in detectedsymb_sentence if item[1] != '']
    followers = [lst[1].split(',') for lst in detectedsymb_sentence]
    followers = unlist(followers)

    # Tickers Symbol matching directly
    # headline_FullSymbMatch = short_tickers_pattern.findall(headline_text)
    # content_FullSymbMatch =
    # headline_FullSymbMatch = re.findall(pattern=short_tickers_pattern, string=headline_text)
    # content_FullSymbMatch = re.findall(pattern=long_tickers_pattern, string=content_text)

    # Merge found tickers from the three algos
    leaders = set(leaders)
    followers = set(followers)
    set_allsymbs = set(tickers_bracket)
    leaders = set_allsymbs.intersection(leaders)
    followers = set_allsymbs.intersection(followers)
    followers = followers - leaders

    print(f"Leaders: {leaders}. Followers: {followers}")
    return timestamp, author, list(leaders), list(followers), content_text, headline_text, detectedsymb_sentence


def leaders_symbols(headline_text):
    # Symbols in brackets
    headline_res = bracket_symbols_extract(text=headline_text)

    # IF can not find tickers within brackets
    # Mapping company name segments to tickers
    if len(headline_res) == 0 and bracket_symbols_only is False:
        headline_res = [seg[1] for seg in seg_pattern if len(seg[0].findall(string=headline_text)) != 0]

    # IF can not fine any tickers within bracket nor company name segments
    # Find the tickers that are not in the brackets, now only applied on headline
    if len(headline_res) == 0 and bracket_symbols_only is False:
        headline_res = unbracket_ticker_matching(text=headline_text, tickers_bracket=tickers_bracket,
                                                 ticker_len_mini=ticker_len_mini)

    return headline_res


def followers_symbols(sentence):
    content_bracket_res = bracket_symbols_extract(text=sentence)
    # remove_bracket_pattern = re.compile(r'\(.*?\)')
    if segment_content_match_flag is True and bracket_symbols_only is False:
        # segments_matched_res = segments_matching(text=sentence, seg_pat_lst=seg_pat_lst)
        segments_matched_res = [seg[1] for seg in seg_pattern if len(seg[0].findall(string=sentence)) != 0]
        content_symbols = list(set(content_bracket_res + segments_matched_res))
    else:
        content_symbols = content_bracket_res

    return ','.join(content_symbols) if len(content_symbols) != 0 else ''


if __name__ == '__main__':
    """
    1. Extract all the NASDAQ and NYSE symbols from all the news content
    2. Extract all the ... from news title
    """
    with open(os.getcwd() + '/Algo_TickersIdentify/SampleNews.pkl', 'rb') as sample_rick:
        SampleNews = pickle.load(sample_rick)

    # ========== Multi-process
    t0 = time.perf_counter()

    parallel_results = []

    with cf.ProcessPoolExecutor() as executor:
        results = executor.map(symbol_extracter,
                               SampleNews.loc[:, 'article_title'].values,
                               SampleNews.loc[:, 'article_content'].values,
                               SampleNews.loc[:, 'article_time'].values,
                               SampleNews.loc[:, 'author_name'].values)

        for result in results:
            parallel_results.append(result)

    t1 = time.perf_counter()

    print(f"Running Time: {round(t1 - t0, 5)}")
    # ======= Multi-process END

    # STORE THE RESULT
    nasdaq_article_tickers = pd.DataFrame(parallel_results,
                                          columns=['datetime', 'author', 'leader', 'follower', 'content', 'headline',
                                                   'detected_sentence'])
    tickers_detected = nasdaq_article_tickers.loc[:, ['datetime', 'leader', 'follower']]

    # ==== save all columns, only tickers, only text
    # to csv
    file_name = '/Algo_TickersIdentify/SampleResult_TickersIdentification'
    tickers_detected.to_csv(os.getcwd() + f'{file_name}.csv')

```

automatically created on 2021-09-17