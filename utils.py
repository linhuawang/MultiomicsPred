import logging
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import normalize
from time import time
from multiprocessing import Pool
from tqdm import trange
import sys
from os.path import exists, abspath
logging.basicConfig(level=logging.INFO)
import os

def get_gloc_from_atac_data(atac_data):
    """Method to get the genomic locations (including the middle point)of peaks"""
    glocs = atac_data.var.index.tolist()
    glocs = [c for c in glocs if 'chr' in c]
    chrms, ranges, sts, ends, midpoints = [], [], [], [], []
    for gl in glocs:
        chrms.append(gl.split('-')[0])
        st, end = int(gl.split('-')[1]), int(gl.split('-')[2])
        sts.append(st)
        ends.append(end)
        midpoints.append(int((st + end)/2))
        ranges.append("_".join(gl.split("-")[1:]))
    gloc_df = pd.DataFrame({'chrm': chrms, 'grange': ranges,
                        'start': sts, 'end': ends,
                        'midpoint': midpoints}, index=glocs)

def get_close_peaks_for_gene(gene, gloc_df,  ref_tss_fn="hg38_ref_TSS.txt",
                        up_dist=5000, down_dist=5000, promoter_dist=2000):
    """Method to get peaks that are within certain range of the TSS site of a gene"""
    ref_tss = pd.read_csv(ref_tss_fn, sep='\t')
    ents = ref_tss.loc[ref_tss.symbol == gene]

    if ents.empty:
        return pd.DataFrame()

    peak_ids = []
    peak_dists = []
    promoters = []

    for i in range(len(ents)):
        chrm = ents.seqnames.tolist()[i]
        st = ents.start.tolist()[i]
        end = ents.end.tolist()[i]
        strand = ents.strand.tolist()[i]
        chrm_gloc = gloc_df.loc[gloc_df.chrm == chrm, :]
        if strand == '+':
            peaks = chrm_gloc.loc[((chrm_gloc['midpoint'] >= st - up_dist) & (chrm_gloc['midpoint'] <= end)) |
                    ((chrm_gloc['midpoint'] <= end + down_dist) & (chrm_gloc['midpoint'] >= end))]
            dists = (peaks.midpoint - st).tolist()

        else:
            peaks = chrm_gloc.loc[((chrm_gloc['midpoint'] <= end + up_dist) & (chrm_gloc['midpoint'] >= st)) |
                ((chrm_gloc['midpoint'] >= st - down_dist) & (chrm_gloc['midpoint'] <= st))]
            dists = (end  - peaks.midpoint).tolist()
        peak_ids += peaks.index.tolist()
        peak_dists += dists
        promoters += [1 if (d >= -2000 and d <= 0) else 0 for d in dists]
    df = pd.DataFrame({"peak": peak_ids, "tss_dist": peak_dists, "pRegion": promoters})
    return df

def embed_mod1(train_mod1, test_mod1):
	"""Method that concatenate the modality one AnnData of the training and testing
		Can be replaced by anndata.concatenate()
	"""
    train_mod1.obs['group'] = 'train'
    test_mod1.obs['group'] = 'test'
    mod1 = train_mod1.concatenate(test_mod1)
    mod1.obs['batch'] = [b.split("-")[-2] for b in mod1.obs.index.tolist()]
    logging.info("### Data merged ####")
    ## select representative features
    return mod1

def read_data(mod1_train_fn, mod1_test_fn, mod2_train_fn):
	"""Helper function to read data for training and testing"""
    train_mod1 = sc.read_h5ad(mod1_train_fn)
    train_mod2 = sc.read_h5ad(mod2_train_fn) 
    test_mod1 = sc.read_h5ad(mod1_test_fn)
    return train_mod1, train_mod2, test_mod1

def dist_matrix_with_feature_downsample(param):
	"""This method samples features and observations of the modality one, 
		then, the sampled data will be embedded to the PCA space and extract
		the neighbors for each testing node.

		Parameters:
		-----------
		mod1: AnnData, embedded modality one.
		seed: int, random seed
		frac_feature: float, proportion of features to be downsampled
		frac_sample: float, proportion of observations to be downsampled
		kneighbor: int, number of neighbors to be extracted for each testing node

		Return:
		------
		pca_simi: pd.DataFrame, columns as testing observations, rows are sampled training
					observations, each entry represents the similarity score
	"""
    mod1, seed, frac_feature, frac_sample, kneighbor = param
    train_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'train')[0]
    test_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'test')[0]
    test_samples = mod1.obs.index[test_idx].tolist()
    train_samples = mod1.obs.index[train_idx].tolist()

    features = mod1.var.index.tolist()
    s = int(frac_feature*len(features))

    np.random.seed(seed)
    sampled_features = np.random.choice(features, size=s, replace=False).tolist()
    
    sampled_train_samples = np.random.choice(train_samples,
        size=int(frac_sample*len(train_samples)), replace=False).tolist()

    sampled_samples = sampled_train_samples + test_samples
    sampled_mod1 = mod1[sampled_samples, sampled_features]
    ## run pca and umap
    t0 = time()
    sc.tl.pca(sampled_mod1, n_comps=50)
    t1 = time()
    logging.info(f"### [{seed}] PCA done in {t1-t0} seconds")

    pca_dist = pdist(sampled_mod1.obsm['X_pca'])
    pca_simi = np.exp(-1 * pca_dist)
    pca_simi = squareform(pca_simi)

    train_idx = np.where(np.ravel(sampled_mod1.obs['group'].to_numpy()) == 'train')[0]
    test_idx = np.where(np.ravel(sampled_mod1.obs['group'].to_numpy()) == 'test')[0]

    pca_simi = pca_simi[test_idx]
    pca_simi = pca_simi[:, train_idx]
    # define neighors
    pca_simi = np.apply_along_axis(func1d=filter_distant_cells,
        axis=1, arr=pca_simi,n=int(kneighbor * frac_sample) + 100)

    pca_simi = pd.DataFrame(data=pca_simi, 
                index=sampled_mod1.obs.index[test_idx],
                columns=sampled_mod1.obs.index[train_idx])
    pca_simi = pca_simi.loc[test_samples,:]
    pca_simi = pca_simi.transpose() # row as train samples and columns as test samples
    pca_simi['sampleID'] = pca_simi.index.tolist()
    return pca_simi

def qc_before_pca(data, top_hvg=0.9, dr_filt=98):
	"""Method to select highly variable genes and drop highly drop-out genes before PCA
	
		Parameters:
		----------
		data: AnnData
		top_hvg: float, 0-1, top percentile of highly variable genes to be kept
		dr_filt: float, 0-100, genes with more than dropout percentage will be filtered

		Return:
		-------
		data: filtered AnnData
	"""
    sc.pp.calculate_qc_metrics(data, inplace=True)
    sc.pp.highly_variable_genes(data, n_top_genes=int(top_hvg*data.shape[1]))
    # exclude a small proportion of conserved genes in PCA analysis
    data = data[:, data.var.index[data.var.highly_variable]]
    # exclude a small proportion of peaks with extremely 
    data = data[:, data.var.index[data.var.pct_dropout_by_counts<dr_filt]]
    return data

def calculate_ensemble_pca_simi(mod1, n_cpu, n_repeats,
    frac_feature, frac_sample, kneighbor):
	"""Method to calculate the similarity between testing nodes and training nodes 
		based on modality 1

		Parameters:
		-----------
		mod1: AnnData, co-embedded modality 1
		n_cpu: int, number of CPUs to be used
		n_repeats: int, number of mini-patches to be used
		frac_feature: float, fraction of features to be downsampled
		frac_sample: float, fraction of observations to be downsampled
		kneigbor: int, number of neigbors to be kept for each testing node

		Return:
		------
		pca_simi: scipy.sparce.csc_matrix, 
					rows are testing nodes, columns are training nodes,
					entry representing similarity score.
	"""
    k_rounds = int(n_repeats / n_cpu)
    logging.info(f"{k_rounds} round using {n_cpu} CPUs and {n_repeats} repeats.")

    train_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'train')[0]
    test_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'test')[0]
    test_samples = mod1.obs.index[test_idx].tolist()
    train_samples = mod1.obs.index[train_idx].tolist()

    pca_simi = []
    for i in range(k_rounds):
        with Pool(processes=n_cpu) as pool:
            st_i = i * n_cpu
            params = [[mod1, st_i + j, frac_feature, frac_sample, kneighbor] for j in range(n_cpu)]
            pca_sims = pool.map(dist_matrix_with_feature_downsample, params)
            pca_sims = pd.concat(pca_sims)
            pca_sims = pca_sims.groupby("sampleID").mean()
            pca_sims.index.name = None
            pca_sims['sampleID'] = pca_sims.index.tolist()
            pca_simi.append(pca_sims)
            logging.info(f"### Round {i+1}/{k_rounds} done.")

    pca_simi = pd.concat(pca_simi)
    pca_simi = pca_simi.groupby("sampleID").mean()
    pca_simi.index.name = None
    pca_simi = pca_simi.transpose()

    other_samples = [sid for sid in train_samples if sid not in pca_simi.columns]
    other_df = pd.DataFrame(data=np.zeros((pca_simi.shape[0], len(other_samples))),
        index=pca_simi.index,
         columns=other_samples)
    pca_simi = pd.concat([pca_simi, other_df], axis=1)


    logging.info("### Similarity matrix aggregated ###")

    vals = np.apply_along_axis(func1d=filter_distant_cells,
        axis=1, arr=pca_simi.values,n=1000)
    #vals = pca_simi.values
    vals = normalize(vals, axis=1, norm='l1')
    pca_simi = pd.DataFrame(data=vals, index=pca_simi.index, columns=pca_simi.columns)
    pca_simi = pca_simi.loc[test_samples, train_samples]
    pca_simi.fillna(0, inplace=True)
    return csc_matrix(pca_simi.values)

def filter_distant_cells(X, n=1000):
    """Helper function to only keep the highest top n values per row.
    """
    ranking = int(np.min([n, 0.01 * len(X)]))
    k = np.sort(X)[::-1][ranking - 1]
    X[X < k] = 0
    return X