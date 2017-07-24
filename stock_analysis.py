# -*- coding: utf-8 -*-

import csv
import os.path
import urllib
import operator
from collections import Counter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, Imputer
from mpl_toolkits.mplot3d import Axes3D

NASDAQ_TICKERS = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download"
NYSE_TICKERS = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&render=download"
YAHOO_STRING = "http://finance.yahoo.com/d/quotes.csv?f=nabb2b3poc1cc6k2p2c8c3ghk1ll1t8w1w4p1mm2kjj5k4j6" \
               "k5wva5b6k3a2ee7e8e9b4j4p5p6rr2r5r6r7s7ydr1qd1d2t1m5m6m7m8m3m4g1g3g4g5g6vj1j3f6n4ss1xj2t7t6i5l2l3v1v7s6&s="
NASDAQ_LIST = 'nasdaq.csv'
NYSE_LIST = 'nyse.csv'
# Yahoo does not allow long query URIs, therefore we limit the max number of allowed tickers
max_entries = 1000
nan_threshold = 0.6
# Column names
column_names = ['Name', 'Ask', 'Bid', 'Ask (Realtime)', 'Bid (Realtime)', 'Previous Close', 'Open',
                'Change', 'Change & Percent Change', 'Change (Realtime)', 'Change Percent (Realtime)',
                'Change in Percent', 'After Hours Change (Realtime)', 'Commission', 'Day\'s Low', 'Day\'s High',
                'Last Trade (Realtime) With Time', 'Last Trade (With Time)', 'Last Trade (Price Only)',
                '1 yr Target Price', 'Day\'s Value Change', 'Day\'s Value Change (Realtime)', 'Price Paid',
                'Day\'s Range', 'Day\'s Range (Realtime)', '52 Week High', '52 week Low',
                'Change From 52 Week Low', 'Change From 52 week High', 'Percent Change From 52 week Low',
                'Percent Change From 52 week High', '52 week Range', 'Volume', 'Ask Size', 'Bid Size',
                'Last Trade Size', 'Average Daily Volume', 'Earnings per Share', 'EPS Estimate Current Year',
                'EPS Estimate Next Year', 'EPS Estimate Next Quarter', 'Book Value', 'EBITDA', 'Price / Sales',
                'Price / Book', 'P/E Ratio', 'P/E Ratio (Realtime)', 'PEG Ratio', 'Price / EPS Estimate Current Year',
                'Price / EPS Estimate Next Year', 'Short Ratio', 'Dividend Yield', 'Dividend per Share',
                'Dividend Pay Date', 'Ex-Dividend Date', 'Last Trade Date', 'Trade Date', 'Last Trade Time',
                'Change From 200 Day Moving Average', 'Percent Change From 200 Day Moving Average',
                'Change From 50 Day Moving Average', 'Percent Change From 50 Day Moving Average',
                '50 Day Moving Average', '200 Day Moving Average', 'Holdings Gain Percent', 'Annualized Gain',
                'Holdings Gain', 'Holdings Gain Percent (Realtime)', 'Holdings Gain (Realtime)', 'More Info',
                'Market Capitalization', 'Market Cap (Realtime)', 'Float Shares', 'Notes', 'Symbol', 'Shares Owned',
                'Stock Exchange', 'Shares Outstanding', 'Ticker Trend', 'Trade Links', 'Order Book (Realtime)',
                'High Limit', 'Low Limit', 'Holdings Value', 'Holdings Value (Realtime)', 'Revenue']
# Whether use cached data
use_cached = True
use_cached_tickers = True
# Max number of clusters to test
max_clusters = 10
# Visualize silhouettes
visualize_silhouettes = False
# Clusterer technique (either 'kmeans' or 'meanshift')
clusterer_method = 'kmeans'

def visualize_with_silhouettes(X, clusterer, cluster_labels, n_clusters, sample_silhouettes):
    # Adopted from scikit documentation
    y_lower = 10
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouettes[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    plt.show()   
    
    
if not use_cached:
    
    def read_tickers(ticker_query, ticker_list):
        tickers = []
        if not use_cached_tickers or (use_cached_tickers and not os.path.isfile(ticker_list)):
            urllib.urlretrieve(ticker_query, ticker_list)
        with open(ticker_list, 'r') as csvfile:
            for row in csv.reader(csvfile, delimiter=',', quotechar='"', dialect=csv.excel):
                tickers.append(row[0])
        return tickers

    tickers = read_tickers(NASDAQ_TICKERS, NASDAQ_LIST)
    tickers += read_tickers(NYSE_TICKERS, NYSE_LIST)
    
    from_ = 0
    to = max_entries
    dataset = pd.DataFrame(columns=column_names)
    while from_ < len(tickers):
        print 'Retrieving entries from %d to %d' % (from_, to)
        query = YAHOO_STRING;
        for ticker in tickers[from_:to]:
            query += ticker + "+";
        query = query.strip("+")
        urllib.urlretrieve(query, 'yahoo_data.csv')
        df = pd.read_csv('yahoo_data.csv', header=False, sep=",", quotechar='"')
        df.columns = column_names
        dataset = pd.concat([dataset, df])
        from_ += max_entries
        to += max_entries
    # Remove entries of tickers which data was not obtained
    print 'Retrieved data of %d tickers total' % len(tickers)
    dataset = dataset[dataset.Name.notnull()]
    # Drop columns with null values
    dataset.dropna(axis=1, how='all', inplace=True)
    
    
    def convert_numeric(x):
        x = str(x)
        value = None
        if x.endswith('M'):
            value = float(x[:-1]) * 1000000
        elif x.endswith('B'):
            value = float(x[:-1]) * 1000000000
        elif x.endswith('K'):
            value = float(x[:-1]) * 1000
        else:
            try:
                value = float(x)
            except:
                return None
        return value / 1000000;
    
    
    # Perform some data cleansing (convert percentages to floats)
    dataset['Change in Percent'] = dataset['Change in Percent'].apply(lambda s: str(s).replace('%', ''))
    dataset['Change in Percent'].astype(float)
    dataset['Percent Change From 52 week Low'] = dataset['Percent Change From 52 week Low'].apply(
        lambda s: str(s).replace("%", ""))
    dataset['Percent Change From 52 week Low'].astype(float)
    dataset['Percent Change From 52 week High'] = dataset['Percent Change From 52 week High'].apply(
        lambda s: str(s).replace('%', ''))
    dataset['Percent Change From 52 week High'].astype(float)
    dataset['Percent Change From 200 Day Moving Average'] = dataset['Percent Change From 200 Day Moving Average'].apply(
        lambda s: str(s).replace('%', ''))
    dataset['Percent Change From 200 Day Moving Average'].astype(float)
    dataset['Percent Change From 50 Day Moving Average'] = dataset['Percent Change From 50 Day Moving Average'].apply(
        lambda s: str(s).replace('%', ''))
    dataset['Percent Change From 50 Day Moving Average'].astype(float)
    # Normalize values by expressing them in millions
    dataset['EBITDA'] = dataset['EBITDA'].apply(convert_numeric)
    dataset['EBITDA'].astype(float)
    dataset['Market Capitalization'] = dataset['Market Capitalization'].apply(convert_numeric)
    dataset['Market Capitalization'].astype(float)
    dataset['Revenue'] = dataset['Revenue'].apply(convert_numeric)
    dataset['Revenue'].astype(float)
    dataset['Shares Outstanding'] = dataset['Shares Outstanding'].apply(lambda s: s / 1000000)
    dataset['Float Shares'] = dataset['Float Shares'].apply(lambda s: s / 1000000)
    # Parse dates as well
    dataset['Dividend Pay Date'] = pd.to_datetime(dataset['Dividend Pay Date'])
    dataset['Ex-Dividend Date'] = pd.to_datetime(dataset['Ex-Dividend Date'])
    dataset['Last Trade Date'] = pd.to_datetime(dataset['Last Trade Date'] + ' ' + dataset['Last Trade Time'])
    # Remove redundant columns
    dataset.drop(['Change & Percent Change', 'Last Trade (With Time)', 'Day\'s Range',
                  '52 week Range', 'More Info', 'Last Trade Time'], axis=1, inplace=True)
    # Save the obtained dataset to CSV
    dataset.to_csv('dataset.csv', header=True, sep=";", quotechar='"')

dataset = pd.read_csv('dataset.csv', header=0, sep=";", quotechar='"',
                      parse_dates=range(42, 46), infer_datetime_format=True, dayfirst=False)
# Form dataset for analysis
andst = dataset[['Change in Percent', '1 yr Target Price', 'Percent Change From 52 week Low',
                 'Percent Change From 52 week High', 'Price / Sales',
                 'Price / Book', 'P/E Ratio', 'PEG Ratio', 'Price / EPS Estimate Current Year',
                 'PEG Ratio', 'Price / EPS Estimate Next Year', 'Short Ratio', 'Dividend per Share',
                 'Change From 200 Day Moving Average', 'Percent Change From 200 Day Moving Average',
                 'Change From 50 Day Moving Average', 'Percent Change From 50 Day Moving Average',
                 '50 Day Moving Average', '200 Day Moving Average']]
andst['Low-High Range'] = dataset['Day\'s High'] - dataset['Day\'s Low']
andst['Volume Ratio'] = dataset['Average Daily Volume'] / dataset['Volume']
andst.loc[np.isinf(andst['Volume Ratio'])] = 0
andst = andst.astype(float)
andst = andst.replace(np.inf, 0)
andst.dropna(axis=1, thresh=round(len(andst) * nan_threshold), inplace=True)
andst.dropna(axis=0, thresh=round(len(andst.columns) * nan_threshold), inplace=True)

# Scale dataset and perform clustering
labels = dataset['Symbol']
data = np.array(andst[:-1])
data = Imputer(missing_values="NaN", strategy="mean", axis=0).fit(data).transform(data)
data = MinMaxScaler().fit(data).transform(data)

pca = decomposition.PCA(n_components=3)
pca.fit(data)
X = pca.transform(data)

if clusterer_method == 'kmeans':
    silhouette_vals = dict()
    for n_clusters in xrange(2, max_clusters+1):    
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_avg)
        silhouette_vals[n_clusters] = silhouette_avg      
        sample_silhouettes = silhouette_samples(data, cluster_labels)
        
        if visualize_silhouettes:
            visualize_with_silhouettes(X, clusterer, cluster_labels, n_clusters, sample_silhouettes)
    
    # Optimal number of clusters is determined by the largest silhouette value. 
    opt_clusters = max(silhouette_vals.iteritems(), key=operator.itemgetter(1))[0]
    print 'Estimated number of clusters:', opt_clusters
    print 'Performing clustering and visualization using the selected number of clusters'
    clusterer = MiniBatchKMeans(n_clusters=opt_clusters, random_state=10)  
    cluster_labels = clusterer.fit_predict(data)
    
elif clusterer_method == 'meanshift':
    bandwidth = estimate_bandwidth(data, n_samples=500)
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    cluster_labels = clusterer.fit_predict(data)
    opt_clusters = len(np.unique(clusterer.labels_))
    print("number of estimated clusters : %d" % opt_clusters)
    
else:
    # Reverting to default clusterer
    opt_clusters = max_clusters
    clusterer = MiniBatchKMeans(n_clusters=opt_clusters, random_state=10) 
    cluster_labels = clusterer.fit_predict(data)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
colors = cm.spectral(cluster_labels.astype(float) / opt_clusters)
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], marker='.', s=30, lw=0, alpha=0.7, c=colors)
ax2.set_title("The visualization of the clustered data, using %d clusters" % opt_clusters)
ax2.set_xlabel("1st component")
ax2.set_ylabel("2nd component")
ax2.set_zlabel("3rd component")
ax2.legend()
plt.show()
fig.savefig('clusters.png', bbox_inches='tight')

# Read company sectors and industries, and combine tem with cluster information
nasdaq_comp = pd.read_csv(NASDAQ_LIST, header=0, sep=",", quotechar='"')
companies = nasdaq_comp[['Symbol', 'Sector', 'Industry']]
nyse_comp = pd.read_csv(NYSE_LIST, header=0, sep=",", quotechar='"')
companies = companies.append(nyse_comp[['Symbol', 'Sector', 'Industry']])

companies_clust = pd.DataFrame.from_records(zip(labels, cluster_labels), 
                                            columns=['Symbol', 'Cluster'])
companies = pd.merge(companies, companies_clust, how='left', on='Symbol')
companies = companies[companies['Cluster'].notnull()]

centroids = clusterer.cluster_centers_
# As we prefer higher values of the ratios in the dataset, we also prefer companies from the cluster with highest 
# values of centroids; thus we weight these clusters according to the number of maximum values
# Recommendation is equal to the weight
maxvals = np.argmax(centroids, axis=0)
maxstats = Counter(maxvals)
weights = dict()
for cl in maxstats.keys():
    weights[cl] = maxstats[cl]/len(maxvals)
companies['Weight'] = companies['Cluster'].apply(lambda x: weights[x]) 

# Draw a heatmap of companies distribution
group = companies.groupby(['Sector', 'Industry'])['Symbol'].count().reset_index()
pivot = group.pivot('Industry', 'Sector', 'Symbol')
sns.set()
fig, ax = plt.subplots()
fig.set_size_inches(0.5 * len(pivot.columns), 0.2 * len(pivot.index))
ax.set_title("Distribution of companies by their industry and sector")
ax.set_xlabel("Sector")
ax.set_ylabel("Industry")
sns.heatmap(pivot, linewidths=.5, ax=ax)
plt.savefig('distribution.png', bbox_inches='tight')

# Draw a heatmap of companies distribution by their recommendation weight
group = companies.groupby(['Sector', 'Industry'])['Weight'].mean().reset_index()
pivot = group.pivot('Industry', 'Sector', 'Weight')
sns.set()
fig, ax = plt.subplots()
fig.set_size_inches(0.5 * len(pivot.columns), 0.2 * len(pivot.index))
ax.set_title("Distribution of companies by mean of recommendations")
ax.set_xlabel("Sector")
ax.set_ylabel("Industry")
sns.heatmap(pivot, linewidths=.5, ax=ax)
plt.savefig('recommendations.png', bbox_inches='tight')