# Dependencies:
# pip: scikit-learn, anndata, scanpy, tqdm, numpy, scipy

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
from utils import *

__author__: Linhua Wang
__email__: linhuaw@bcm.edu

def atac2gex(train_mod1, test_mod1, train_mod2, n_cpu, n_repeats, 
                frac_feature, frac_sample, kneighbor, ref_tss_fn):
    """
    Method that predicts GEX from ATAC

    Parameters:
    -----------
    train_mod1: AnnData of the ATAC training data
    train_mod2: AnnData of the GEX training data, same .obs as train_mod1
    test_mod1: AnnData of the ATAC testing data, same.var as train_mod1
    n_cpu: int, number of cpu to be used, number <= 10 if memory limited.
    n_repeats: int, the number of mini batches to be used, suggested number <= 50
    frac_feature: float, 0-1, porportion of features to be sampled in each model
    frac_sample: float, 0-1, proportion of observation to be sampled in each model
    ref_tss_fn: str, path to the TSS reference file.

    Return:
    -------
    predicted_adata: AnnData of predicted GEX on test data
    """
    t0 = time()
    mod1 = embed_mod1(train_mod1, test_mod1)
    t1 = time()
    #---------------------- Select peaks based on their distance to TSS -------------#
    gloc_df = get_gloc_from_atac_data(mod1)
    genes = train_mod2.var.index.tolist()
    logging.info(f"{mod1.shape[1]} peaks before extracting.")
    peaks = []
    k = 1
    for gene in genes:
        peak_df = get_close_peaks_for_gene(gene, gloc_df, ref_tss_fn,
                up_dist=5000, down_dist=5000, promoter_dist=2000)
        if not peak_df.empty:
            peaks += peak_df.peak.tolist()
        if k  % 1000 == 1:
            logging.info(f"{k}/{len(genes)} genes processed.")
        k += 1
    peaks = list(set(peaks))
    logging.info(f"{len(peaks)} tss associated peaks after extracted.")
    mod1 = mod1[:, peaks]
    #---------------------- Select highly variable peaks -----------------------#
    logging.info(f"###[ATAC2GEX] Data merged in {t1-t0:.2f} seconds ###")
    logging.info(f"###[ATAC2GEX] {mod1.shape[1]} peaks before filtering ###")
    mod1 = qc_before_pca(mod1, top_hvg=0.8)
    logging.info(f"###[ATAC2GEX] {mod1.shape[1]} peaks after filtering ###")

    #---------------------- Now working on binary peak derived PCA -----------------------#
    t2 = time()
    pca_simi = calculate_ensemble_pca_simi(mod1, n_cpu, n_repeats, 
        frac_feature, frac_sample, kneighbor) 

    t5 = time()
    logging.info(f"###[ATAC2GEX] Weight matrices calculated in {t5-t2:.2f} seconds ###")

    train_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'train')[0]
    test_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'test')[0]
    train_samples = mod1.obs.index[train_idx].tolist()
    test_samples = mod1.obs.index[test_idx].tolist()
    train_mod2_samples = ["-".join(t.split("-")[:-1]) for t in train_samples]
    train_mod2 = train_mod2[train_mod2_samples,:]
    test_mod1_samples = ["-".join(t.split("-")[:-1]) for t in test_samples]

    #---------------------- Assign zero to extremly high dropout genes ---------------------#
    sc.pp.calculate_qc_metrics(train_mod2, inplace=True)
    genes = train_mod2.var.index[train_mod2.var.pct_dropout_by_counts > 98]
    train_filtered_mod2 = train_mod2.copy()
    train_filtered_mod2[:, genes] = 0
    X = pca_simi * train_filtered_mod2.X
#    X = pca_simi * train_mod2.X
    X = csc_matrix(X)
    predict_adata = ad.AnnData(X=X, obs=test_mod1.obs.loc[test_mod1_samples,:], var=train_mod2.var)
    return predict_adata


def gex2atac(train_mod1, test_mod1, train_mod2, 
            n_cpu, n_repeats, frac_feature, frac_sample, kneighbor):
    """
    Method that predicts ATAC from GEX

    Parameters:
    -----------
    train_mod1: AnnData of the GEX training data
    train_mod2: AnnData of the ATAC training data, same .obs as train_mod1
    test_mod1: AnnData of the GEX testing data, same.var as train_mod1
    n_cpu: int, number of cpu to be used, number <= 10 if memory limited.
    n_repeats: int, the number of mini batches to be used, suggested number <= 50
    frac_feature: float, 0-1, porportion of features to be sampled in each model
    frac_sample: float, 0-1, proportion of observation to be sampled in each model

    Return:
    -------
    predicted_adata: AnnData of predicted ATAC on test data
    """
    t0 = time()
    mod1 = embed_mod1(train_mod1, test_mod1)
    t1 = time()
    logging.info(f"###[GEX2ATAC] Data merged in {t1-t0:.2f} seconds ###")
    logging.info(f"###[GEX2ATAC] {mod1.shape[1]} genes before filtering ###")
    mod1 = qc_before_pca(mod1, top_hvg=0.6, dr_filt=100)
    logging.info(f"###[GEX2ATAC] {mod1.shape[1]} genes after filtering ###")

    #---------------------- Now working on gene expression derived PCA -----------------------#
    logging.info(f"###[GEX2ATAC] Multiprocessing to get ensembled weight matrices ###")
    pca_simi = calculate_ensemble_pca_simi(mod1, n_cpu, n_repeats,
        frac_feature, frac_sample)

    t2 = time()
    logging.info(f"###[GEX2ATAC] Weight matrix ({pca_simi.shape[0]}, {pca_simi.shape[1]}) calculatedd in {t2-t1} seconds###")

    train_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'train')[0]
    test_idx = np.where(np.ravel(mod1.obs['group'].to_numpy()) == 'test')[0]
    train_samples = mod1.obs.index[train_idx].tolist()
    test_samples = mod1.obs.index[test_idx].tolist()    
    train_mod2_samples = ["-".join(t.split("-")[:-1]) for t in train_samples]
    train_mod2 = train_mod2[train_mod2_samples,:]
    test_mod1_samples = ["-".join(t.split("-")[:-1]) for t in test_samples]

    #---------------------- Assign zero to extremly high dropout peaks ---------------------#
    X = pca_simi * train_mod2.X
    X = csc_matrix(X)
    predict_adata = ad.AnnData(X=X, obs=test_mod1.obs.loc[test_mod1_samples,:], var=train_mod2.var)
    return predict_adata

def main(args):
    st = time()
    mode = args.m
    mod1_train_fn, mod2_train_fn, mod1_test_fn = args.a, args.b, args.c
    n_cpu, n_repeats = args.n, args.N
    frac_sample, frac_feature, kneighbor = args.s, args.f, args.k

    if mode == "atac2gex":
        gex2atac(train_mod1, test_mod1, train_mod2, 
            n_cpu, n_repeats, frac_feature, frac_sample, kneighbor)
    else:
        reference_tss = args.r
        atac2gex(train_mod1, test_mod1, train_mod2,
            n_cpu, n_repeats, frac_feature, frac_sample, kneighbor, reference_tss)
    end = time()
    logging.info(f"Task {mode} completed in {end-st} seconds.")
    return predict_adata

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train_mod1', metavar='a', type=str,
                    help='training of modality 1')
    parser.add_argument('--input_train_mod2', metavar='b', type=str,
                    help='training of modality 2')
    parser.add_argument('--input_test_mod1', metavar='c', type=str,
                    help='testing of modality 1')
    parser.add_argument('--output_test_mod2', metavar='o', type=str,
                    help='testing of modality 1')
    parser.add_argument('--ref_tss_fn', metavar='r', type=str,
                    help='testing of modality 1')
    parser.add_argument('--mode', metavar='m', type=str,
                    help='atac2gex or gex2atac')
    parser.add_argument('--n_cpu', metavar='n', type=int,
                    help='number of cpu to be used', default=2)
    parser.add_argument('--n_repeats', metavar='N', type=int,
                    help='number of cpu to be used', default=20)
    parser.add_argument('--kneighbor', metavar='k', type=int,
                    help='number of cpu to be used', default=20)
    parser.add_argument('--frac_sample', metavar='s', type=float,
                    help='number of cpu to be used', default=0.5)
    parser.add_argument('--frac_feature', metavar='f', type=float,
                    help='number of cpu to be used', default=0.3)

    args = parser.parse_args()
    output_mod2 = main(args)
    output_mod2.write_h5ad(args.o, compression='gzip')