import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def get_df_residue(df):
    def normalise_df(df):
        df_mean = df.mean(axis = 0).values
        df_std = df.std(axis = 0).values
        df_normalised = (df - df_mean) / df_std
        return df_normalised, df_mean, df_std
    rows, features = df.shape
    df_normalised, df_mean, df_std = normalise_df(df)
    pca = PCA(n_components = features)
    pca.fit(df_normalised)
    v0 = pca.components_[0]
    
    projection = np.outer(df_normalised @ v0, v0)
    projection = (projection * df_std) + df_mean
    df_res = df - projection
    
    df_eigenvectors = pca.components_
    df_eigenvalues = pca.explained_variance_
    return df_res, df_eigenvectors, df_eigenvalues

def get_PCA_analysis(df,window = 604800, step = 302400):
    def get_df_stats(df):
        df_stats = df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        df_stats['normalized_mean'] = df_stats.eval('mean / std')
        df_stats['skew'] = df.skew()
        df_stats['kurtosis'] = df.kurtosis()
        return df_stats
    residues = []
    stats = []
    eigenvectors = []
    eigenvalues = []
    rows, features = df.shape

    for left in range(0, rows - window, step):
        y = df[left : left + window]
        y_res, y_eigenvectors, y_eigenvalues = get_df_residue(y)
        residues.append(y_res)
        stats.append(get_df_stats(y_res))
        eigenvectors.append(y_eigenvectors)
        eigenvalues.append(y_eigenvalues)
        
    v0 = []
    v0value = []
    def normalise_vector(v, position):
        u = v / v[position]
        return u
    for i in range(len(eigenvectors)):
        v0.append(normalise_vector(eigenvectors[i][0], 0))########################here we are normalising the eigenvector by letting the 1st element to be 1
        v0value.append(eigenvalues[i][0])

    eigenvector_df = pd.DataFrame({'Eigenvector' : v0, 'Eigenvalue' : v0value})
    temp_stat_df = pd.concat(stats, keys = range(len(stats)))
    temp_stat_df.index.names = ['window', 'feature']
    temp_stat_df = temp_stat_df.swaplevel().sort_index()
    return {'residues' : residues, 'stats' : temp_stat_df, 'eigenvectors' : eigenvectors, 'eigenvalues' : eigenvalues, 'eigenvector_df' : eigenvector_df}
