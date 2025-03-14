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
import pyspark.sql.functions as f
from pyspark.sql.types import FloatType, StringType
from pyspark.sql import DataFrame

# Create similarity function with a target vector
def create_similarity_function(spark, movie_id: int, method: str = 'cosine', results: DataFrame = None):
    # Method validation
    if method not in ('cosine', 'jaccard'):
        print('Error: only "cosine" or "jaccard" methods available.')
        return None
    
    # Movie selector validation
    try:
        # Get title and vector representation
        movie = results.\
            filter(results.id == movie_id).\
            select('title', 'features').collect()[0]

        # Show movie name
        print(f'Chosen movie: {movie[0]}.')

        # Broadcast (cache) movie vector for quick access
        bc_mv = spark.sparkContext.broadcast(movie[1])

    except IndexError:
        print(f'Error: no movie with the index {movie_id} was found.')
        return None

    # Chose method to compute similarity
    if method == 'cosine':
        def sim_computation(other_movie):
            # Formula: sim = A·B /|A||B|
            num = float(np.dot(bc_mv.value, other_movie))
            
            mod_a = float(np.sqrt(np.dot(bc_mv.value, bc_mv.value)))
            mod_b = float(np.sqrt(np.dot(other_movie, other_movie)))
            den = mod_a * mod_b

            # If either A or B is an empty string, there is
            # nothing to compare, thus return no similarity (0)
            return num / den if den != 0 else 0.0
    
    else: # jaccard similarity
        def sim_computation(other_movie):
            # Formula: Sim = intersection(A,B) / Union(A,B)
            A_and_B = float(sum(1 for (a,b) in zip(bc_mv.value,other_movie) if a==1 and a==b)) # Intersection
            A_or_B = float(sum(1 for (a,b) in zip(bc_mv.value,other_movie) if a==1 or b==1)) # Union
            
            # If either A or B is an empty string, there is
            # nothing to compare, thus return no similarity (0)
            return A_and_B / A_or_B if A_or_B != 0 else 0.0 # Similarity

    return f.udf(lambda vec: sim_computation(vec), FloatType())



