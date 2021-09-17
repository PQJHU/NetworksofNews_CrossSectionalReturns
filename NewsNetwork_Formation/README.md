[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **NewsNetwork** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : NewsNetwork

Published in : 'Networks of News and Cross-Sectional Returns'

Description : 'Construct news network out of the identified tickers of S&P500 stocks'

Keywords : Textual Analysis, Network Analysis

See also : ''

Author : Junjie Hu

```

### PYTHON Code
```python

"""
Python: 3.8.10
Given the results of identified tickers from news, the script construct the network/connection between the companies
"""

import pandas as pd
import networkx as nx
import datetime as dt
import itertools
import matplotlib.pyplot as plt
import pickle
import os
import ast
import re
import numpy as np
import math
import concurrent.futures as cf
import time
import calendar


def count_weighted_edges(edges_list):
    """
    Count the edges and return list of edges with frequency of appearance
    :param edges_list:
    :return:
    """
    # Count edges frequency
    count_edges = pd.value_counts(edges_list)
    count_edges = count_edges.to_frame()
    count_edges.reset_index(inplace=True)
    weighted_edges = count_edges.values.tolist()
    weighted_edges = [(item[0][0], item[0][1], item[1]) for item in weighted_edges]
    return weighted_edges


def adj_row_normalize(adj_df):
    """
    Given a adjacency matrix, row nor
    :param adj_df:
    :return:
    """
    adj_df_out = adj_df.copy(deep=True)
    for row in adj_df_out.index:
        adj_df_out.loc[row, :] = adj_df_out.loc[row, :] / adj_df_out.loc[row, :].sum()
    adj_df_out.fillna(0, inplace=True)
    adj_df_out = adj_df_out[adj_df_out.index]
    return adj_df_out


def networks_construction_periodic(g_spx_w, nodes_list, dir_label, start, end):
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    print(f'From {start_date} To {end_date}')

    edges_list = g_spx_w.values.tolist()
    adj_dir = os.path.join(os.getcwd(), 'NewsNetwork_Formation/AdjacencyMatrix/')

    """
    Weighted Network
    """
    undirected_edges_list = [sorted(edge) for edge in edges_list]
    weighted_edges = count_weighted_edges(undirected_edges_list)
    # Weighted Undirected Graph
    G_ud_weighted = nx.Graph()
    for node in nodes_list.values:
        G_ud_weighted.add_node(node)

    G_ud_weighted.add_weighted_edges_from(weighted_edges)
    G_ud_weighted = G_ud_weighted.subgraph(nodes_list)  # Keep only the firms that are in the current spx constituents

    uw_adj_dir = os.path.join(adj_dir, f'{dir_label}/')
    os.makedirs(uw_adj_dir, exist_ok=True)

    ud_w_adj_df = pd.DataFrame(index=G_ud_weighted.nodes, columns=G_ud_weighted.nodes,
                               data=np.transpose(nx.convert_matrix.to_numpy_matrix(G_ud_weighted)))

    ud_w_adj_df = adj_row_normalize(ud_w_adj_df)
    ud_w_adj_df.to_csv(uw_adj_dir + f"{start_date}_{end_date}.csv")

    # Weighted Directed Graph
    direced_weighted_edges = count_weighted_edges(edges_list)

    G_d_weighted = nx.DiGraph()
    for node in nodes_list.values:
        G_d_weighted.add_node(node)

    G_d_weighted.add_weighted_edges_from(direced_weighted_edges)
    G_d_weighted = G_d_weighted.subgraph(nodes_list)  # Keep only the firms that are in the current spx constituents

    d_w_adj_df = pd.DataFrame(index=G_d_weighted.nodes, columns=G_d_weighted.nodes,
                              data=np.transpose(nx.convert_matrix.to_numpy_matrix(G_d_weighted)))
    d_w_adj_df = adj_row_normalize(d_w_adj_df)

    dw_adj_dir = os.path.join(adj_dir, f'{dir_label}/')
    os.makedirs(dw_adj_dir, exist_ok=True)
    d_w_adj_df.to_csv(dw_adj_dir + f"{start_date}_{end_date}.csv")


def OneTypeNetworkConstruction(network_fullsample, dir_label, full_nodes):
    network_fullsample.loc[:, 'date'] = [date.date() for date in network_fullsample.index]
    network_fullsample.set_index('date', drop=True, inplace=True)
    years = [2016]  # specify the years in the sample
    month_params = list(itertools.product(years, range(1, 13)))
    seg_ts = [(dt.date(month_param[0], month_param[1], 1),
               dt.date(month_param[0], month_param[1], calendar.monthrange(month_param[0], month_param[1])[1])
               ) for month_param in month_params]
    network_chunks = [network_fullsample.loc[(network_fullsample.index >= chunkdate[0]) &
                                             (network_fullsample.index <= chunkdate[1])] for chunkdate in seg_ts]
    nodes_chunks = [full_nodes.loc[full_nodes.index.date == chunkdate[1].replace(day=1), 'Ticker'] for chunkdate in
                    seg_ts]

    dir_labels = [dir_label] * len(seg_ts)
    start_dates = [segdate[0] for segdate in seg_ts]
    end_dates = [segdate[1] for segdate in seg_ts]

    start_timer = time.perf_counter()

    with cf.ProcessPoolExecutor() as executor:
        executor.map(networks_construction_periodic,
                     network_chunks,
                     nodes_chunks,
                     dir_labels,
                     start_dates,
                     end_dates)
    networks_construction_periodic(g_spx_w=network_chunks[0], nodes_list=nodes_chunks[0], dir_label=dir_label, start=start_dates[0], end=end_dates[0])
    end_timer = time.perf_counter()
    # Full Sample Network
    full_nodes_list = full_nodes.drop_duplicates().loc[:, 'Ticker']
    networks_construction_periodic(g_spx_w=network_fullsample, nodes_list=full_nodes_list, dir_label='FullSample_'+dir_label, start=start_dates[0], end=end_dates[-1])
    print(f'Finished network {dir_label} in {round(end_timer - start_timer, 2)} Seconds')


if __name__ == '__main__':
    spx_symbols = pd.read_csv(os.getcwd() + '/NewsNetwork_Formation/SPX_TickersEvolv_NameSector.csv', index_col=0,
                              parse_dates=True)

    with open(os.getcwd() + '/NewsNetwork_Formation/SPX_InfluencePairs_Sample.pkl', 'rb') as tickers_rick:
        TickerPairsSample = pickle.load(tickers_rick)

    # Round the time frame. Count the articles before 9am on day t+1 to t.
    # Equivalently, shifting the time frame 9-hours back
    TickerPairsSample['trading_date'] = [dtime - dt.timedelta(hours=9) for dtime in TickerPairsSample.index]
    TickerPairsSample.set_index('trading_date', inplace=True)

    mix_sector_network = TickerPairsSample.loc[:, ['leader', 'follower']]
    inter_sector_network = TickerPairsSample.loc[TickerPairsSample['common_sector'] == False, ['leader', 'follower']]
    intra_sector_network = TickerPairsSample.loc[TickerPairsSample['common_sector'] == True, ['leader', 'follower']]

    OneTypeNetworkConstruction(network_fullsample=mix_sector_network, dir_label='All Sectors',
                               full_nodes=spx_symbols['Ticker'].to_frame())
    OneTypeNetworkConstruction(network_fullsample=inter_sector_network, dir_label='Cross-Sector',
                               full_nodes=spx_symbols['Ticker'].to_frame())
    OneTypeNetworkConstruction(network_fullsample=intra_sector_network, dir_label='Within-Sector',
                               full_nodes=spx_symbols['Ticker'].to_frame())

```

automatically created on 2021-09-17