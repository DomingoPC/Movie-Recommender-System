import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

def load_and_show(path: str, spark_session, parquet=False):
    '''
    Read parquet files with header and quotations with "
    '''
    # Load file with " as quotation marks
    read_options = spark_session.read.\
        option('quote', '\"').\
        option('escape', '\"').\
        option('header', True).\
        option('inferSchema', True)

    if parquet:
        df = read_options.parquet(path)
    else:
        df = read_options.csv(path)
    
    # Columns data types
    print('Columns data types:')
    display(pd.DataFrame(df.dtypes, columns = ['Column Name', 'Data Type']))

    # Number of partitions and number of entries
    print(f'Number of partitions = {df.rdd.getNumPartitions()}')
    print(f'Number of entries/rows = {df.count()}\n')

    # Print first 5 entries
    print('Data sample:')
    display(df.limit(5).toPandas())

    # Descriptions
    print('Data description:')
    display(df.describe().toPandas())
    return df

def show_correlation(df):
    '''
    Calculates correlation between numeric variables. Then, plot the correlation matrix.
    '''

    # Compute the correlation matrix
    corr = df.select_dtypes(include='number').corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5)

    plt.tight_layout()
    plt.show()

    return corr

# --- TF-IDF ---




