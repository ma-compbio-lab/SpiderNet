##############################################################################################################################
# Package imports
from torch_scatter import scatter_mean
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import MiniBatchNMF
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import time
from pathlib import Path
import scipy.sparse as sp
from scipy.stats import rankdata
cuda_available = torch.cuda.is_available()
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    ##
    device = "cuda"
else:
    print("No GPU available, using CPU instead.")
    device = "cpu"

def compute_coexpression(edge_index, exp_use_sparse, crosscorrelation_choose1):
    # 1. Construct edge_index_hstack and its reversed version
    edge_index_hstack = edge_index[:, 0]
    edge_index_hstack_reversed = edge_index[:, 1]

    # 2. Extract target and neighboring expressions
    exp_target = exp_use_sparse[:, crosscorrelation_choose1[0]]
    selector_matrix = sp.csr_matrix(
        (np.ones_like(edge_index_hstack), (np.arange(len(edge_index_hstack)), edge_index_hstack)),
        shape=(len(edge_index_hstack), exp_target.shape[0]),
        dtype=np.int16
    )
    exp_target_extend = selector_matrix.dot(exp_target)
    del selector_matrix, exp_target

    exp_neigh = exp_use_sparse[:, crosscorrelation_choose1[1]]
    selector_matrix = sp.csr_matrix(
        (np.ones_like(edge_index_hstack_reversed),
         (np.arange(len(edge_index_hstack_reversed)), edge_index_hstack_reversed)),
        shape=(len(edge_index_hstack_reversed), exp_neigh.shape[0]),
        dtype=np.int16
    )
    exp_neigh_extend = selector_matrix.dot(exp_neigh)
    del selector_matrix, exp_neigh

    # 3. Compute co-expression values
    exp_coexpression_extend = np.sqrt(exp_target_extend.multiply(exp_neigh_extend))

    return exp_coexpression_extend

def spearman_corr_matrix(A, B):
    # Step 1: Rank values within each row (axis=1)
    A_ranked = np.apply_along_axis(rankdata, 1, A)
    B_ranked = np.apply_along_axis(rankdata, 1, B)

    # Step 2: Compute the Pearson correlation matrix (equivalent to Spearman)
    # Center each row
    A_centered = A_ranked - A_ranked.mean(axis=1, keepdims=True)
    B_centered = B_ranked - B_ranked.mean(axis=1, keepdims=True)

    # Normalize each row
    A_norm = A_centered / np.linalg.norm(A_centered, axis=1, keepdims=True)
    B_norm = B_centered / np.linalg.norm(B_centered, axis=1, keepdims=True)

    # Compute similarity via dot product
    spearman_matrix = np.dot(A_norm, B_norm.T)
    return spearman_matrix

def Initial_model(SpiderNet_data_pyg_list,dim_envir,
                  Factor_mode,dim_intri = None,n_jobs = 10,Initial_regression = "Linear",
                  enhance_init_with_gene_coexp = True,threshold_crosscorr = 0.3,numtop_crosscorr = 10,spearcorr_use_rowmax_threshold = 0.3):
    edgenum_batch = np.cumsum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'].shape[0] for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list))])
    edgenum_batch_extend = np.hstack((0, edgenum_batch))
    ##Intrinsic part
    Y_use = np.vstack([SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['x'].cpu().numpy() for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list))])
    if Factor_mode == "NMF":
        ##NMF initialization
        nmf_intrinsic = NMF(n_components=dim_intri, init='nndsvda', random_state=0, max_iter=1000)
        factor_nmf_intrinsic = nmf_intrinsic.fit_transform(Y_use)
        factor_intrinsic_init = factor_nmf_intrinsic
        ##Normalization
        factor_intrinsic_init_colmax = np.max(factor_intrinsic_init, axis=0)
        factor_intrinsic_init = factor_intrinsic_init / factor_intrinsic_init_colmax
    else:
        factor_intrinsic_init = np.vstack([SpiderNet_data_pyg_list[batch_cell_unique_cur_index][Factor_mode + '_onehot'].cpu().numpy() for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list))])
    ##Initial loading_intrinsic_init
    loading_intrinsic_init = []
    for dim_intri_index in range(factor_intrinsic_init.shape[1]):
        loading_intrinsic_init.append(np.mean(Y_use[np.where(factor_intrinsic_init[:, dim_intri_index] == 1)[0],:],axis = 0))
    loading_intrinsic_init = np.array(loading_intrinsic_init)
    ##
    Residual = Y_use - np.matmul(factor_intrinsic_init, loading_intrinsic_init)
    ##Turn the residual to be non-negative
    Residual[Residual < 0] = 0
    ##Environmental LR part
    ##Obtain the cell-paired Ligand-Receptor expression
    cellpair_LRpair_neigh = np.vstack([SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['cellpair_LRpair_neigh'].cpu().numpy() for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list))])
    cellpair_LRpair_neigh_batch = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        edge_index_start = int(edgenum_batch_extend[batch_cell_unique_cur_index])
        edge_index_end = int(edgenum_batch_extend[batch_cell_unique_cur_index + 1])
        cellpair_LRpair_neigh_batch.append(cellpair_LRpair_neigh[edge_index_start:edge_index_end, :])
    # exp_use = Residual
    exp_use = Y_use.copy()
    exp_use_batch = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        cell_index_start = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index)]))
        cell_index_end = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index + 1)]))
        exp_use_batch.append(exp_use[cell_index_start:cell_index_end, :])
    ##
    factor_intrinsic_init_index = [np.where(factor_intrinsic_init[i,:] == 1)[0][0] for i in range(factor_intrinsic_init.shape[0])]
    factor_intrinsic_init_index = np.array(factor_intrinsic_init_index)
    factor_intrinsic_init_index_unique = np.unique(factor_intrinsic_init_index)
    factor_intrinsic_init_index_batch = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        cell_index_start = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index)]))
        cell_index_end = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index + 1)]))
        factor_intrinsic_init_index_batch.append(
            factor_intrinsic_init_index[cell_index_start:cell_index_end])
    factor_intrinsic_init_batch = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        cell_index_start = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index)]))
        cell_index_end = int(np.sum([SpiderNet_data_pyg_list[batch_cell_unique_cur_index2]['num_cells'] for batch_cell_unique_cur_index2 in range(batch_cell_unique_cur_index + 1)]))
        factor_intrinsic_init_batch.append(
            factor_intrinsic_init[cell_index_start:cell_index_end, :])
    ##
    if enhance_init_with_gene_coexp:
        # print("Co-expression-based initialization.")
        crosscorrelation_choose1_batch = []
        for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
            edge_index_sender = SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'].to("cpu").numpy()[
                                :, 0]
            edge_index_receiver = SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'].to(
                "cpu").numpy()[:, 1]
            edge_index_receiver_sender = np.vstack((edge_index_receiver, edge_index_sender)).T
            crosscorrelation_choose1_list = []
            if edge_index_receiver_sender.shape[0] > 1e+6:
                sample_ratio = 5
                np.random.seed(123)
            else:
                sample_ratio = 1
            for cellclass_receiver_index in factor_intrinsic_init_index_unique:
                # print("Receiver class index: " + str(cellclass_receiver_index))
                edge_index_receiver_sender_cur0 = edge_index_receiver_sender[
                                                  factor_intrinsic_init_index_batch[batch_cell_unique_cur_index][
                                                      edge_index_receiver_sender[:,
                                                      0]] == cellclass_receiver_index, :]
                for cellclass_sender_index in factor_intrinsic_init_index_unique:
                    edge_index_receiver_sender_cur = edge_index_receiver_sender_cur0[
                                                     factor_intrinsic_init_index_batch[batch_cell_unique_cur_index][
                                                         edge_index_receiver_sender_cur0[:,
                                                         1]] == cellclass_sender_index, :]
                    if edge_index_receiver_sender_cur.shape[0] > 50:
                        ##sample edge_index_receiver_sender_cur
                        num_edge_cur = edge_index_receiver_sender_cur.shape[0]
                        if sample_ratio > 1:
                            choose_index = np.random.choice(num_edge_cur, int(num_edge_cur / sample_ratio),
                                                            replace=False)
                            edge_index_receiver_sender_cur = edge_index_receiver_sender_cur[choose_index, :]
                        # print("Receiver class index: " + str(cellclass_receiver_index) + " Sender class index: " + str(cellclass_sender_index))
                        ##
                        edge_index_receiver_sender_cur0_unique = np.unique(edge_index_receiver_sender_cur[:, 0])
                        edge_index_receiver_sender_cur1_unique = np.unique(edge_index_receiver_sender_cur[:, 1])
                        exp_use_sub_receiver = exp_use[edge_index_receiver_sender_cur0_unique, :]
                        exp_use_sub_sender = exp_use[edge_index_receiver_sender_cur1_unique, :]
                        geneexp_mean_receiver = np.mean(exp_use_sub_receiver, axis=0)
                        geneexp_mean_sender = np.mean(exp_use_sub_sender, axis=0)
                        geneexp_std_receiver = np.std(exp_use_sub_receiver, axis=0)
                        geneexp_std_sender = np.std(exp_use_sub_sender, axis=0)
                        # geneexp_std_trans = geneexp_std * math.sqrt(exp_use_sub.shape[0])
                        geneexp_std_receiver_trans = geneexp_std_receiver.reshape(-1, 1).T
                        geneexp_std_sender_trans = geneexp_std_sender.reshape(-1, 1).T
                        expression_normalized_receiver = exp_use_sub_receiver - geneexp_mean_receiver
                        expression_normalized_sender = exp_use_sub_sender - geneexp_mean_sender
                        geneexp_std_trans2 = np.matmul(geneexp_std_receiver_trans.T, geneexp_std_sender_trans)
                        geneexp_std_trans2[geneexp_std_trans2 == 0] = 1
                        ##
                        edge_index_receiver_sender_cur_rel = [
                            [np.where(edge_index_receiver_sender_cur0_unique == edge_index_receiver_sender_cur[i, 0])[
                                 0][0],
                             np.where(edge_index_receiver_sender_cur1_unique == edge_index_receiver_sender_cur[i, 1])[
                                 0][0]]
                            for i in range(edge_index_receiver_sender_cur.shape[0])]
                        edge_index_receiver_sender_cur_rel = np.array(edge_index_receiver_sender_cur_rel)
                        expression_normalized_centered_target = expression_normalized_receiver[
                                                                edge_index_receiver_sender_cur_rel[:, 0], :]
                        expression_normalized_centered_neigh = expression_normalized_sender[
                                                               edge_index_receiver_sender_cur_rel[:, 1], :]
                        ##
                        crosscovariance = np.matmul(expression_normalized_centered_target.T,
                                                    expression_normalized_centered_neigh)
                        crosscovariance = crosscovariance * (1 / expression_normalized_centered_target.shape[0])
                        crosscorrelation = crosscovariance / geneexp_std_trans2
                        ##
                        # threshold_complex = threshold_crosscorr
                        threshold_complex = max(threshold_crosscorr, np.quantile(crosscorrelation, 1 - (
                                numtop_crosscorr / (crosscorrelation.shape[0] * crosscorrelation.shape[1]))))
                        crosscorrelation_choose1_cur = np.where(crosscorrelation >= threshold_complex)
                        crosscorrelation_choose1_list.append(crosscorrelation_choose1_cur)
            crosscorrelation_choose1_list_merge = np.hstack(crosscorrelation_choose1_list)
            crosscorrelation_choose1_list_merge_unique = np.unique(crosscorrelation_choose1_list_merge, axis=1)
            ##
            crosscorrelation_choose1 = (
                crosscorrelation_choose1_list_merge_unique[0, :], crosscorrelation_choose1_list_merge_unique[1, :])
            # crosscorrelation_choose1 = np.where(np.abs(crosscorrelation) >= threshold_crosscorr)
            # crosscorrelation_choose2 = np.where(crosscorrelation <= -threshold_crosscorr)
            # print("The number of selected gene pair: " + str(crosscorrelation_choose1[0].shape[0]))
            crosscorrelation_choose1_batch.append(crosscorrelation_choose1)
        crosscorrelation_choose1 = np.unique(np.hstack(
            [np.array(crosscorrelation_choose1_batch[batch_cell_unique_cur_index]) for batch_cell_unique_cur_index in
             range(len(crosscorrelation_choose1_batch))]), axis=1)
        crosscorrelation_choose1 = (crosscorrelation_choose1[0, :], crosscorrelation_choose1[1, :])
        ##
        exp_coexpression_extend_list = []
        for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
            edge_index_cpu = SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'].to("cpu").numpy()
            edge_index_index = np.array(range(edge_index_cpu.shape[0]))
            # if edge_index_cpu.shape[0] > 1e+6:
            #     edge_index_index = np.random.choice(edge_index_index, int(1e+6), replace=False)
            #     ##randomly sample the edge_index_cpu
            #     edge_index_cpu = edge_index_cpu[edge_index_index, :]
            exp_use_np = np.array(exp_use_batch[batch_cell_unique_cur_index]).astype(np.float32)
            # exp_use_sparse = csr_matrix(exp_use_np)
            exp_use_sparse = sp.csr_matrix(exp_use_np)
            exp_coexpression_extend = compute_coexpression(edge_index_cpu, exp_use_sparse, crosscorrelation_choose1)
            exp_coexpression_extend_list.append(exp_coexpression_extend)

        ##Positive group's coexpression
        if crosscorrelation_choose1[0].shape[0] > 0:
            spearcorr_use_rowmax_curbatch_list = []
            for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
                edge_index_cpu = SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'].to("cpu").numpy()
                edge_index_index = np.array(range(edge_index_cpu.shape[0]))
                ##
                x = exp_coexpression_extend_list[batch_cell_unique_cur_index].toarray().T
                y = SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['cellpair_LRpair_neigh'].T.to("cpu").numpy()
                if x.shape[1] > 1e+5:
                    sample_index_cur = np.random.choice(np.array(range(x.shape[1])), int(1e+5), replace=False)
                    spearcorr_use = spearman_corr_matrix(x[:, sample_index_cur], y[:, sample_index_cur])
                else:
                    spearcorr_use = spearman_corr_matrix(x, y)
                spearcorr_use_rowmax_curbatch = np.max(spearcorr_use, axis=1)
                spearcorr_use_rowmax_curbatch_list.append(spearcorr_use_rowmax_curbatch)
            ##
            spearcorr_use_rowmax = np.max(np.vstack(spearcorr_use_rowmax_curbatch_list), axis=0)
            spearcorr_use_rowmax_index = spearcorr_use_rowmax > spearcorr_use_rowmax_threshold
            # print(np.unique(spearcorr_use_rowmax_index, return_counts=True))
            ##
            # if edge_index_index.shape[0] == SpiderNet_data_pyg['edge_index'].shape[0]:
            #     exp_coexpression_extend = sp.vstack(exp_coexpression_extend_list)
            #     exp_coexpression_extend = exp_coexpression_extend.toarray()
            #     exp_coexpression_extend = exp_coexpression_extend[:, spearcorr_use_rowmax_index]
            # else:
            #     edge_index_cpu = SpiderNet_data_pyg['edge_index'].to("cpu").numpy()
            #     crosscorrelation_choose1_choose = (crosscorrelation_choose1[0][spearcorr_use_rowmax_index], crosscorrelation_choose1[1][spearcorr_use_rowmax_index])
            #     ##
            #     exp_coexpression_extend = compute_coexpression(edge_index_cpu, exp_use_sparse, crosscorrelation_choose1_choose)
            #     exp_coexpression_extend = exp_coexpression_extend.toarray()
            # exp_coexpression_extend = sp.vstack(exp_coexpression_extend_list)
            # exp_coexpression_extend = exp_coexpression_extend[:, spearcorr_use_rowmax_index]
            # exp_coexpression_extend = exp_coexpression_extend.toarray()
            # exp_coexpression_extend = sp.vstack([
            #     m[:, spearcorr_use_rowmax_index] for m in exp_coexpression_extend_list
            # ]).toarray()
            # ##
            # exp_coexpression_extend_LR = np.hstack((cellpair_LRpair_neigh, exp_coexpression_extend))
            spearcorr_use_rowmax_index1 = np.where(spearcorr_use_rowmax_index)[0]
            n_blocks = len(exp_coexpression_extend_list)
            n_rows_list = [m.shape[0] for m in exp_coexpression_extend_list]
            n_cols_right = len(spearcorr_use_rowmax_index1)
            # print("spearcorr_use_rowmax_index1 length" + str(n_cols_right))
            n_rows_total = sum(n_rows_list)
            n_cols_left = cellpair_LRpair_neigh.shape[1]

            # Preallocate the combined matrix
            exp_coexpression_extend_LR = np.empty(
                (n_rows_total, n_cols_left + n_cols_right),
                dtype=np.float32  # Match the input dtype
            )

            # Fill the matrix block by block
            row_start = 0
            for i, m in enumerate(exp_coexpression_extend_list):
                sub = m[:, spearcorr_use_rowmax_index1].toarray()  # Convert the selected block to dense
                n_rows = sub.shape[0]
                exp_coexpression_extend_LR[row_start:row_start + n_rows, :n_cols_left] = cellpair_LRpair_neigh[
                                                                                         row_start:row_start + n_rows]
                exp_coexpression_extend_LR[row_start:row_start + n_rows, n_cols_left:] = sub
                row_start += n_rows
            # exp_coexpression_extend_LR = cellpair_LRpair_neigh
        else:
            exp_coexpression_extend_LR = cellpair_LRpair_neigh
        del exp_coexpression_extend
    else:
        exp_coexpression_extend_LR = cellpair_LRpair_neigh
    # exp_coexpression_extend_LR = cellpair_LRpair_neigh
    # print("NMF Start.")
    ##NMF on exp_coexpression_extend_LR
    if exp_coexpression_extend_LR.shape[0]>5000000:
        exp_coexpression_extend_LR_sparse = sp.csr_matrix(exp_coexpression_extend_LR)
        model_NMF_exp_coexpression_extend_LR = MiniBatchNMF(n_components=dim_envir, init='nndsvda', random_state=0,
                                                            # batch_size=int(exp_coexpression_extend_LR.shape[0]/50),
                                                            batch_size=max(
                                                                int(exp_coexpression_extend_LR.shape[0] / 100), 50000),
                                                            max_iter=5000)
        factor_GP_minibatchNMF = model_NMF_exp_coexpression_extend_LR.fit_transform(exp_coexpression_extend_LR_sparse)
        # factor_GP_minibatchNMF = model_NMF_exp_coexpression_extend_LR.fit_transform(exp_coexpression_extend_LR)
        del exp_coexpression_extend_LR_sparse
        loading_LR_init = model_NMF_exp_coexpression_extend_LR.components_
    else:
        model_NMF_exp_coexpression_extend_LR = NMF(n_components=dim_envir, init='nndsvda', random_state=0,
                                                   max_iter=3000)
        factor_GP_minibatchNMF = model_NMF_exp_coexpression_extend_LR.fit_transform(exp_coexpression_extend_LR)
        loading_LR_init = model_NMF_exp_coexpression_extend_LR.components_
    # exp_coexpression_extend_LR_sparse = sp.csr_matrix(exp_coexpression_extend_LR)
    # model_NMF_exp_coexpression_extend_LR = MiniBatchNMF(n_components=dim_envir, init='nndsvda', random_state=0,
    #                                                     # batch_size=int(exp_coexpression_extend_LR.shape[0]/50),
    #                                                     batch_size=max(
    #                                                         int(exp_coexpression_extend_LR.shape[0] / 100), 50000),
    #                                                     # max_iter=3000)
    #                                                     max_iter=1000)
    # ##
    # factor_GP_minibatchNMF = model_NMF_exp_coexpression_extend_LR.fit_transform(exp_coexpression_extend_LR_sparse)
    # del exp_coexpression_extend_LR_sparse
    # loading_LR_init = model_NMF_exp_coexpression_extend_LR.components_
    # print("NMF finish.")
    ##Normalization
    factor_GP_minibatchNMF_colmax = np.max(factor_GP_minibatchNMF, axis=0)
    factor_GP_minibatchNMF = factor_GP_minibatchNMF / factor_GP_minibatchNMF_colmax
    ##filter
    # factor_GP_minibatchNMF[factor_GP_minibatchNMF<0.2] = 0
    # factor_GP_minibatchNMF = np.power(factor_GP_minibatchNMF, 2)
    factor_GP_minibatchNMF[factor_GP_minibatchNMF < 0.5] = np.power(
        factor_GP_minibatchNMF[factor_GP_minibatchNMF < 0.5], 2)
    loading_LR_init = loading_LR_init * factor_GP_minibatchNMF_colmax[:, None]
    # ##Way1
    # loading_LR_init = loading_LR_init[:, 0:cellpair_LRpair_neigh.shape[1]]
    ##Way2
    # cellpair_LRpair_neigh
    # factor_GP_minibatchNMF
    # print("LR regression start.")
    reg_ls_loading_LR = LinearRegression(fit_intercept=False, positive=True, n_jobs=n_jobs)
    if factor_GP_minibatchNMF.shape[0] > 1000000:
        # print("downsampling")
        sampleindex = np.random.choice(range(factor_GP_minibatchNMF.shape[0]), max(int(factor_GP_minibatchNMF.shape[0]/5),1000000), replace=False)
    else:
        sampleindex = np.array(range(factor_GP_minibatchNMF.shape[0]))
    reg_res_loading_LR = reg_ls_loading_LR.fit(factor_GP_minibatchNMF[sampleindex,:], cellpair_LRpair_neigh[sampleindex, :])
    # print("LR regression end.")
    ##
    loading_LR_init = (reg_res_loading_LR.coef_).copy().T
    ##
    factor_GP_minibatchNMF_batch = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        edge_index_start = int(edgenum_batch_extend[batch_cell_unique_cur_index])
        edge_index_end = int(edgenum_batch_extend[batch_cell_unique_cur_index + 1])
        factor_GP_minibatchNMF_batch.append(factor_GP_minibatchNMF[edge_index_start:edge_index_end, :])
    factor_envir_agg_list = []
    for batch_cell_unique_cur_index in range(len(SpiderNet_data_pyg_list)):
        factor_GP_minibatchNMF_curbatch = factor_GP_minibatchNMF_batch[batch_cell_unique_cur_index]

        ## Aggregation
        factor_receiver_agg = scatter_mean(torch.tensor(factor_GP_minibatchNMF_curbatch).to(device),
                                           torch.tensor(
                                               SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'][:,
                                               1]).to(torch.int64).to(device), dim=0,
                                           dim_size=SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['num_cells'])
        factor_sender_agg = scatter_mean(torch.tensor(factor_GP_minibatchNMF_curbatch).to(device),
                                         torch.tensor(
                                             SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['edge_index'][:,
                                             0]).to(torch.int64).to(device), dim=0,
                                         dim_size=SpiderNet_data_pyg_list[batch_cell_unique_cur_index]['num_cells'])
        ##
        factor_sender_agg = factor_sender_agg.cpu().numpy()
        factor_receiver_agg = factor_receiver_agg.cpu().numpy()

        ##
        factor_envir_agg_curbatch = np.hstack((factor_receiver_agg, factor_sender_agg))
        factor_envir_agg_list.append(factor_envir_agg_curbatch)
    factor_envir_agg = np.vstack(factor_envir_agg_list)
    #
    ##
    if factor_envir_agg.shape[0] > 100000:
        # print("downsampling")
        sampleindex = np.random.choice(range(factor_envir_agg.shape[0]), max(int(factor_envir_agg.shape[0]/5),100000), replace=False)
    else:
        sampleindex = np.array(range(factor_envir_agg.shape[0]))
    # print("gene regression start.")
    if Initial_regression == "Linear":
        reg_ls_loading_share = LinearRegression(fit_intercept=False, positive=True, n_jobs=n_jobs)
        reg_res_loading_share = reg_ls_loading_share.fit(factor_envir_agg[sampleindex, :], Residual[sampleindex, :])
    else:
        reg_lasso_loading_share = Lasso(alpha=0.01, fit_intercept=False, positive=True, max_iter=10000)
        reg_res_loading_share = reg_lasso_loading_share.fit(factor_envir_agg[sampleindex, :], Residual[sampleindex, :])
    # print("gene regression end.")
    ##
    loading_receiver_init_share = (reg_res_loading_share.coef_).copy().T[0:dim_envir, :]
    loading_sender_init_share = (reg_res_loading_share.coef_).copy().T[dim_envir:, :]
    ##
    Initial_dict = {"factor_intrinsic_init": factor_intrinsic_init_batch,
                    "loading_intrinsic_init": loading_intrinsic_init,
                    "factor_GP_minibatchNMF": factor_GP_minibatchNMF_batch,
                    "loading_receiver_init": loading_receiver_init_share,
                    "loading_sender_init": loading_sender_init_share,
                    "loading_LR_init": loading_LR_init}
    return Initial_dict

class SpiderNet_model(torch.nn.Module):
    def __init__(self,
                 num_gene,
                 num_LR,
                 hidden_channels,
                 Factor_mode,
                 dim_intri = 10,
                 dim_envir = 10):
        super(SpiderNet_model, self).__init__()
        ##
        self.num_gene = num_gene
        self.num_LR = num_LR
        self.hidden_channels = hidden_channels
        # self.hidden_channels_scale = int(hidden_channels/4)
        self.hidden_channels_scale = int(hidden_channels / 2)
        self.Factor_mode = Factor_mode
        self.dim_intri = dim_intri
        self.dim_envir = dim_envir
        ##
        self.dropout = 0.1
        self.dropout_fun = nn.Dropout(p=self.dropout)
        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        ## Intrinsic factor encoder (if cell type one-hot is not provided, use NMF to learn intrinsic factors)
        if self.Factor_mode == "NMF":
            self.enc_factor_intrinsic = nn.Sequential(
                nn.Linear(num_gene, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, dim_intri),
                nn.Sigmoid()
            )

        ## Environmental encoders
        scale_factor = 2
        # self.enc_factor_envir_pre_receiver = nn.Sequential(
        #     nn.Linear(num_gene, 2 * scale_factor * self.hidden_channels_scale),
        #     nn.ReLU(),
        #     nn.Linear(2 * scale_factor * self.hidden_channels_scale, scale_factor * self.hidden_channels_scale),
        #     nn.ReLU()
        # )
        self.enc_factor_envir_pre_receiver = nn.Sequential(
            nn.Linear(num_gene, 2 * scale_factor * self.hidden_channels_scale),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, scale_factor * self.hidden_channels_scale),
            # nn.ReLU()
            nn.Tanh()
        )
        # self.enc_factor_envir_pre_sender = nn.Sequential(
        #     nn.Linear(num_gene, 2 * scale_factor * self.hidden_channels_scale),
        #     nn.ReLU(),
        #     nn.Linear(2 * scale_factor * self.hidden_channels_scale, scale_factor * self.hidden_channels_scale),
        #     nn.ReLU()
        # )
        self.enc_factor_envir_pre_sender = nn.Sequential(
            nn.Linear(num_gene, 2 * scale_factor * self.hidden_channels_scale),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, scale_factor * self.hidden_channels_scale),
            # nn.ReLU()
            nn.Tanh()
        )
        self.enc_factor_envir = nn.Sequential(
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, 2 * scale_factor * self.hidden_channels_scale),
            nn.Tanh(),
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, self.dim_envir)
        )

        ## learnable loading parameters
        self.Loading_intrinsic_ori = nn.Parameter(torch.abs(torch.randn(self.dim_intri, self.num_gene)))
        self.loading_receiver_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_gene)))
        self.loading_sender_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_gene)))
        self.loading_LR_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_LR)))

    def forward(self, SpiderNet_data_pyg):
        with autocast():
            exp = SpiderNet_data_pyg.x
            num_cell = exp.shape[0]
            edge_index = SpiderNet_data_pyg['edge_index']

            ## Intrinsic factor
            if self.Factor_mode == "NMF":
                Factor_intrinsic = self.enc_factor_intrinsic(exp)
                Factor_intrinsic = Factor_intrinsic / Factor_intrinsic.sum(dim=1, keepdim=True).clamp(min=1e-8)
            else:
                Factor_intrinsic = SpiderNet_data_pyg[self.Factor_mode + '_onehot']

            ## Environmental factors
            exp_enc_receiver = self.enc_factor_envir_pre_receiver(exp)
            exp_enc_sender = self.enc_factor_envir_pre_sender(exp)
            exp_enc_center_receiver = exp_enc_receiver[edge_index[:, 1], :]
            exp_enc_neighbor_sender = exp_enc_sender[edge_index[:, 0], :]
            exp_enc_neighbor_sender_center_receiver = torch.cat(
                (exp_enc_neighbor_sender, exp_enc_center_receiver), dim=1
            )
            Factor_envir_sender_receiver = self.enc_factor_envir(exp_enc_neighbor_sender_center_receiver)
            Factor_envir_sender_receiver = self.Sigmoid(Factor_envir_sender_receiver)
            Factor_envir = Factor_envir_sender_receiver
            edge_receiver = edge_index[:, 1]
            edge_sender = edge_index[:, 0]
            ## Neighborhood aggregation
            Factor_envir_receiver_neighagg = scatter_mean(Factor_envir_sender_receiver, edge_receiver, dim=0,
                                                          dim_size=num_cell)
            Factor_envir_sender_neighagg = scatter_mean(Factor_envir_sender_receiver, edge_sender, dim=0,
                                                        dim_size=num_cell)

            ## Multi-modal reconstruction
            relu_fn = self.Relu
            Loading_intrinsic = relu_fn(self.Loading_intrinsic_ori)
            loading_receiver = relu_fn(self.loading_receiver_ori)
            loading_sender = relu_fn(self.loading_sender_ori)
            loading_LR = relu_fn(self.loading_LR_ori)

            ## Expression reconstruction of different parts
            exp_recon_intrinsic = torch.matmul(Factor_intrinsic, Loading_intrinsic)
            exp_recon_envir_receiver = torch.matmul(Factor_envir_receiver_neighagg, loading_receiver)
            exp_recon_envir_sender = torch.matmul(Factor_envir_sender_neighagg, loading_sender)
            # exp_recon = exp_recon_intrinsic + exp_recon_envir_receiver + exp_recon_envir_sender
            exp_recon = exp_recon_intrinsic
            exp_recon.add_(exp_recon_envir_receiver).add_(exp_recon_envir_sender)
            ## LR reconstruction
            exp_LR_recon = torch.matmul(Factor_envir, loading_LR)

            return exp_recon, exp_LR_recon, Factor_intrinsic, Loading_intrinsic, Factor_envir, loading_receiver, loading_sender, loading_LR

    # def compute_loss_in_batches(self, SpiderNet_data_pyg_list, factor_envir_init, stage, criterion, scaler=None,
    #                             loss_weight=1.0):
    #     """Compute loss batch by batch and accumulate gradients"""
    #     num_batches = len(SpiderNet_data_pyg_list)
    #     loss_total = 0.0
    #
    #     for batch_index_cur in range(num_batches):
    #         exp_recon_cur, exp_LR_recon_cur, Factor_intrinsic_cur, Loading_intrinsic_cur, \
    #             Factor_envir_cur, loading_receiver_cur, loading_sender_cur, loading_LR_cur = self(SpiderNet_data_pyg_list[batch_index_cur])
    #
    #         if stage == 'main_exp':
    #             with autocast():
    #                 target = SpiderNet_data_pyg_list[batch_index_cur].x
    #                 if exp_recon_cur.dtype != target.dtype:
    #                     target = target.to(exp_recon_cur.dtype)
    #                 loss_batch = criterion(exp_recon_cur, target) * (1 / num_batches)
    #
    #         elif stage == 'main_LR':
    #             with autocast():
    #                 target = SpiderNet_data_pyg_list[batch_index_cur]['cellpair_LRpair_neigh']
    #                 if exp_LR_recon_cur.dtype != target.dtype:
    #                     target = target.to(exp_LR_recon_cur.dtype)
    #                 loss_batch = criterion(exp_LR_recon_cur, target) * (1 / num_batches)
    #
    #         else:  # warmup
    #             with autocast():
    #                 target = factor_envir_init[batch_index_cur]
    #                 if Factor_envir_cur.dtype != target.dtype:
    #                     target = target.to(Factor_envir_cur.dtype)
    #                 loss_batch = criterion(Factor_envir_cur, target) * (1 / num_batches)
    #
    #         loss_batch = loss_batch * loss_weight
    #
    #         if scaler is not None:
    #             scaler.scale(loss_batch).backward()
    #         else:
    #             loss_batch.backward()
    #
    #         loss_total += loss_batch.item()
    #
    #     return loss_total

    def compute_loss_in_batches(self, SpiderNet_data_pyg_list, factor_envir_init, stage, criterion, scaler=None,
                                loss_weight=1.0):
        """
        Compute loss batch by batch and accumulate gradients.
        Now supports two stages: 'warmup' and 'main'.
        Returns loss_warmup, loss_exp, loss_LR
        """
        num_batches = len(SpiderNet_data_pyg_list)
        loss_total = 0.0
        loss_warmup, loss_exp, loss_LR = 0.0, 0.0, 0.0

        for batch_index_cur in range(num_batches):
            # Forward pass
            exp_recon_cur, exp_LR_recon_cur, Factor_intrinsic_cur, Loading_intrinsic_cur, \
                Factor_envir_cur, loading_receiver_cur, loading_sender_cur, loading_LR_cur = \
                self(SpiderNet_data_pyg_list[batch_index_cur])

            if stage == 'warmup':
                with autocast():
                    target = factor_envir_init[batch_index_cur]
                    if Factor_envir_cur.dtype != target.dtype:
                        target = target.to(Factor_envir_cur.dtype)
                    loss_batch = criterion(Factor_envir_cur, target) * (1 / num_batches)
                loss_warmup += loss_batch.item()

                if scaler is not None:
                    scaler.scale(loss_batch).backward()
                else:
                    loss_batch.backward()

            elif stage == 'main':
                with autocast():
                    # expression reconstruction
                    target_exp = SpiderNet_data_pyg_list[batch_index_cur].x
                    if exp_recon_cur.dtype != target_exp.dtype:
                        target_exp = target_exp.to(exp_recon_cur.dtype)
                    loss_exp_batch = criterion(exp_recon_cur, target_exp) * (1 / num_batches)

                    # LR reconstruction
                    target_LR = SpiderNet_data_pyg_list[batch_index_cur]['cellpair_LRpair_neigh']
                    if exp_LR_recon_cur.dtype != target_LR.dtype:
                        target_LR = target_LR.to(exp_LR_recon_cur.dtype)
                    loss_LR_batch = criterion(exp_LR_recon_cur, target_LR) * (1 / num_batches)

                    # combine
                    loss_batch = loss_exp_batch + loss_LR_batch * loss_weight

                # backward
                if scaler is not None:
                    scaler.scale(loss_batch).backward()
                else:
                    loss_batch.backward()

                loss_exp += loss_exp_batch.item()
                loss_LR += loss_LR_batch.item()

            loss_total += loss_batch.item()

        return loss_warmup, loss_exp, loss_LR

    def fit(self, SpiderNet_data_pyg_list,
            device: str = 'cuda',
            optim_type: str = 'adam',
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            LR_loss_weight: float = 1,
            warmup=10,
            max_epoch: int = 100,
            loss_fn: str = 'mse',
            Initial_dict=None,
            file_savepath_model_main=None):
        """Training function"""

        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == 'poisson':
            criterion = nn.PoissonNLLLoss(log_input=False)
        else:
            raise ValueError(f"Unsupported loss_fn {loss_fn}")

        scaler = GradScaler()

        loading_params_names = ['Loading_intrinsic_ori', 'loading_receiver_ori', 'loading_sender_ori', 'loading_LR_ori']
        init_params = {
            'Loading_intrinsic_ori': torch.tensor(Initial_dict['loading_intrinsic_init']).to(device).to(torch.float32),
            'loading_receiver_ori': torch.tensor(Initial_dict['loading_receiver_init']).to(device).to(torch.float32),
            'loading_sender_ori': torch.tensor(Initial_dict['loading_sender_init']).to(device).to(torch.float32),
            'loading_LR_ori': torch.tensor(Initial_dict['loading_LR_init']).to(device).to(torch.float32)
        }
        factor_envir_init = [
            torch.tensor(Initial_dict['factor_GP_minibatchNMF'][batch_index_cur]).to(device).to(torch.float32)
            for batch_index_cur in range(len(Initial_dict['factor_GP_minibatchNMF']))]

        all_params = dict(self.named_parameters())
        enc_factor_intrinsic_params = [p for n, p in all_params.items() if "enc_factor_intrinsic" in n]
        enc_factor_envir_params = [p for n, p in all_params.items() if "enc_factor_envir" in n]
        loading_params_trainable = [p for n, p in all_params.items() if n in loading_params_names]

        starttime_top100epoch = time.time()
        for epoch in range(max_epoch):
            self.train()

            if epoch == 0:
                opt_args = dict(lr=5e-4, weight_decay=weight_decay)
                if optim_type == "adam":
                    optimizer = optim.Adam(self.parameters(), **opt_args)
                elif optim_type == 'SGD':
                    optimizer = optim.SGD(self.parameters(), **opt_args)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-5)
            else:
                if epoch == warmup:
                    # print("Start main part.")
                    opt_args = dict(lr=lr, weight_decay=weight_decay)
                    if optim_type == "adam":
                        optimizer = optim.Adam(self.parameters(), **opt_args)
                    elif optim_type == 'SGD':
                        optimizer = optim.SGD(self.parameters(), **opt_args)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-5)

            for name, param in self.named_parameters():
                param.requires_grad = True
            if epoch < warmup:
                for name, param in self.named_parameters():
                    if name in loading_params_names:
                        param.requires_grad = False
                        param.data.copy_(init_params[name].to(param.dtype))

            ## Blocked update
            if epoch >= warmup:
                if self.Factor_mode == "NMF":
                    if epoch % 3 == 0:
                        for p in enc_factor_intrinsic_params:
                            p.requires_grad = True
                        for p in enc_factor_envir_params + loading_params_trainable:
                            p.requires_grad = False
                    elif epoch % 3 == 1:
                        for p in enc_factor_envir_params:
                            p.requires_grad = True
                        for p in enc_factor_intrinsic_params + loading_params_trainable:
                            p.requires_grad = False
                    else:
                        for p in loading_params_trainable:
                            p.requires_grad = True
                        for p in enc_factor_intrinsic_params + enc_factor_envir_params:
                            p.requires_grad = False
                else:
                    # if epoch % 3 < 2:
                    if epoch % 2 < 1 or epoch < warmup + 10:
                        for p in enc_factor_envir_params:
                            p.requires_grad = False
                        for p in loading_params_trainable:
                            p.requires_grad = True
                    else:
                        for p in loading_params_trainable:
                            p.requires_grad = False
                        for p in enc_factor_envir_params:
                            p.requires_grad = True

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                self.train()
                ## Determine which parameters of the model need to be updated
                ## Compute loss
                if epoch < warmup:
                    loss_factor_envir,_,_ = self.compute_loss_in_batches(
                        SpiderNet_data_pyg_list,
                        factor_envir_init,
                        stage='warmup',
                        criterion=criterion,
                        scaler=scaler,
                        loss_weight=1
                    )

                    loss_factor_intrinsic = 0
                    if self.Factor_mode == "NMF":
                        Loss_whole_use = loss_factor_intrinsic + loss_factor_envir
                    else:
                        Loss_whole_use = loss_factor_envir

                    # if epoch % 100 == 0:
                    if epoch == 0 or epoch == warmup - 1:
                        if self.Factor_mode == "NMF":
                            print("Epoch: " + str(epoch) + " train_loss: " + str(
                                Loss_whole_use) + " loss_factor_intrinsic: " + str(
                                loss_factor_intrinsic) + " loss_factor_envir: " + str(
                                loss_factor_envir))
                        else:
                            print("Epoch: " + str(epoch) + " train_loss: " + str(
                                Loss_whole_use) + " loss_factor_intrinsic: " + str(0) + " loss_factor_envir: " + str(
                                loss_factor_envir))
                else:
                    ## Gene expression reconstruction loss
                    # Loss_exprecon = self.compute_loss_in_batches(
                    #     SpiderNet_data_pyg_list,
                    #     factor_envir_init,
                    #     stage='main_exp',
                    #     criterion=criterion,
                    #     scaler=scaler,
                    #     loss_weight=1
                    # )
                    #
                    # Loss_expLRpair_recon = self.compute_loss_in_batches(
                    #     SpiderNet_data_pyg_list,
                    #     factor_envir_init,
                    #     stage='main_LR',
                    #     criterion=criterion,
                    #     scaler=scaler,
                    #     loss_weight=LR_loss_weight
                    # )
                    loss_warmup,Loss_exprecon,Loss_expLRpair_recon = self.compute_loss_in_batches(
                        SpiderNet_data_pyg_list,
                        factor_envir_init,
                        stage='main',
                        criterion=criterion,
                        scaler=scaler,
                        loss_weight=LR_loss_weight
                    )

                    Loss_whole_use = Loss_exprecon + Loss_expLRpair_recon
                    # if epoch % 1000 == 0 or epoch < warmup + 10:
                    if epoch % 1000 == 0:
                        print("Epoch: " + str(epoch) + " train_loss: " + str(
                            Loss_whole_use) + "Loss_exprecon:" + str(
                            Loss_exprecon) + "Loss_expLRpair_recon:" + str(Loss_expLRpair_recon))

            # After accumulating all batches, update parameters once
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()  # Clear gradients to prepare for the next epoch

            # Update the scheduler every 10 epochs after warmup
            if epoch >= warmup and epoch % 10 == 0:
                scheduler.step(Loss_whole_use)

            if epoch % 2000 == 0:
                loss_whole_record = Loss_whole_use

            # Save the model periodically
            if epoch % 1000 == 0 or epoch == max_epoch - 1:
                save_dir = Path(file_savepath_model_main)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), save_dir / f"model_epoch{epoch}.pth")

            # torch.cuda.empty_cache()

