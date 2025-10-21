# Copyright (c) 2025 Yunru Wang
# Based on code by Samuel Valenzuela <Samuel.valenzuela@ifi.lmu.de>; modified by Yunru Wang.
# This file contains modifications and extensions to the original work.
import csv
from pathlib import Path
import argparse
import collections
import functools
import itertools
import math
import os
import pickle
import shlex
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from datasets import load_from_disk

import numpy as np
import pandas as pd
import scipy
import tqdm
from joblib import Parallel, delayed
import json
import gc

from xfl.config import Config
from xfl.evaluation import Evaluation
from xfl.nlp import NLP


class PfastreXML:

    # copied from original
    @staticmethod
    def count_tokens(it_tok_it):
        config = Config()
        nlp = NLP(config)

        canonical_set = []
        c = collections.Counter()
        for tok_it in it_tok_it:
            cs = nlp.canonical_set(tok_it)
            c.update(cs)
            canonical_set.append(cs)
        return canonical_set, c

    @staticmethod
    def name_vector(canonical_sets, label_space):
        def to_vec(token_vector, arr):
            vec = np.zeros((len(token_vector),), dtype=np.int64)
            for it in arr:
                if it in token_vector:
                    vec[token_vector.index(it)] += 1
            return vec

        return [to_vec(label_space, x) for x in canonical_sets]

    @staticmethod
    def train_pfastrexml(pfast_model_path, data_dir, trn_X_fname, trn_Y_fname, a=1.0, b=1.0, c=1.0,
            max_inst_in_leaves=10, l=100, g=30, T=32,
            trees=256):
        """
            Runs PfatsreXML training and test commands with the given hyperparameters
            Sample Usage :
                ./PfastreXML_train [feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 100
                -g 30 -a 0.8 -q 1

                -S PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=0
                -T Number of threads to use. default=1
                -s Starting tree index. default=0
                -t Number of trees to be grown. default=50
                -b Feature bias value, extre feature value to be appended. default=1.0
                -c SVM weight co-efficient. default=1.0
                -m Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves. default=10
                -l Number of label-probability pairs to retain in a leaf. default=100
                -g gamma parameter appearing in tail label classifiers. default=30
                -a Trade-off parameter between PfastXML and tail label classifiers. default=0.8
                -q quiet option (0/1). default=0
                feature and label files are in sparse matrix format
        """
        print('Running new PfastreXML instance under data directory: ', data_dir)
        trn_cmd = pfast_model_path + ' {}/{} {}/{} {}/inv_prop.txt {}/xml_model '.format(data_dir, trn_X_fname,
                data_dir, trn_Y_fname,
                data_dir, data_dir)
        trn_cmd += '-q 0 -S 0 -T {} -t {} -a {} -b {} -c {} -m {} -g {} -l {}'.format(T, trees, a, b, c,
                max_inst_in_leaves, g, l)
        model_dir = '{}/xml_model'.format(data_dir)
        if os.path.exists(model_dir):
            # Empty directory contents
            print('Clearing previous model...')
            shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)

        print('Running: ', trn_cmd)
        res = subprocess.call(shlex.split(trn_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res < 0:
            raise RuntimeError('Error calling PfastreXML subprocess during training')

    @staticmethod
    def pred_pfastrexml(pfast_model_path, data_dir, tst_X_fname):
        tst_cmd = pfast_model_path + ' {}/{} {}/xml_score.mat {}/xml_model'.format(data_dir, tst_X_fname, data_dir,
                data_dir)
        print('Running: ', tst_cmd)
        res = subprocess.call(shlex.split(tst_cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res < 0:
            raise RuntimeError('Error calling PfastreXML subprocess during prediction')

    def __init__(self, config, temp_dir):
        """
            XML classifier
        """
        self.config = config
        self.nlp = NLP(config)
        self.directory = temp_dir
        self.label_space = None
        self.name_df = None
        self.real_name_df = None
        self.canon_name_df = None
        self.canonical_sets = None
        self.X = None
        self.Y = None
        self.trn_X = None
        self.val_X = None
        self.tst_X = None
        self.trn_Y = None
        self.val_Y = None
        self.tst_Y = None
        self.inv_props = None

        # XFL defined a subset of 'freebies' to simulate a real-life scenario in which some functions could be resolved
        # by pattern matching
        self.calculable_knowns = {'init', 'fini', 'csu_init', 'csu_fini', 'start', 'libc_csu_init', 'libc_csu_fini',
                'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main',
                'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry',
                'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start',
                'start_main', 'libc_start_main'}

    def from_data(self, df, k):
        print('Generating label space for top {} labels...'.format(k))
        self.label_space = self.generate_label_space(df['name'].values, k=k)

        print('Parsing embeddings and labels...')
        self.generate_dataframe(df)

        print('Calculating inverse propensities...')
        self.inv_propensities(A=0.5, B=0.425)
        print('Saving to XML txt format...')
        self.save_xml_data()

        # save pandas dataframes
        for k in tqdm.tqdm(['{}_{}'.format(b, e) for b in ('trn', 'val', 'tst') for e in ('X', 'Y')] + ['name_df'],
                desc='Saving model data'):
            value = getattr(self, k)
            value.to_pickle('{}/{}.pickle'.format(self.directory, k))

    def generate_dataframe(self, df):
        """
            Generate a pandas dataframe given a list of
            function names and a list of their embeddings
        """
        names = df['name'].values
        embeddings = df['embedding'].values
        real_names = df['real_name'].values
        canon_names = df['canon_name'].values
        l_index = df.index

        self.name_df = pd.DataFrame(names, columns=['name'], index=l_index)
        self.name_df.to_pickle('{}/name_df.pickle'.format(self.directory))

        embeddings_df = pd.DataFrame(embeddings, columns=['embedding'], index=l_index)
        self.real_name_df = pd.DataFrame(real_names, columns=['real_name'], index=l_index)
        self.canon_name_df = pd.DataFrame(canon_names, columns=['canon_name'], index=l_index)

        # Need to regenerate dataset if label size changes
        # Label size needs to change, otherwise propensities of labels differs
        while True:
            chunks = n_chunks(self.canonical_sets, 256)
            results = Parallel(n_jobs=self.config.analysis.THREAD_POOL_THREADS, verbose=1, backend='multiprocessing')(
                    map(delayed(PfastreXML.name_vector), chunks, itertools.repeat(self.label_space)))
            labels = functools.reduce(lambda x, y: x + y, results, [])
            labels_mat = np.vstack(labels)

            self.X = embeddings_df
            self.Y = pd.DataFrame(data=labels_mat, index=l_index)
            self.trn_X = self.X[self.X.index.isin(df[df['data_split'] == 'train'].index)]
            self.val_X = self.X[self.X.index.isin(df[df['data_split'] == 'train_val'].index)]
            self.tst_X = self.X[self.X.index.isin(df[df['data_split'] == 'test'].index)]
            self.trn_Y = self.Y[self.Y.index.isin(df[df['data_split'] == 'train'].index)]
            self.val_Y = self.Y[self.Y.index.isin(df[df['data_split'] == 'train_val'].index)]
            self.tst_Y = self.Y[self.Y.index.isin(df[df['data_split'] == 'test'].index)]

            n_labels = len(self.label_space)
            # Apply dataset preconditioning/prefiltering
            self.trn_X, self.trn_Y, self.label_space = self.precondition_dataset(self.trn_X, self.trn_Y, self.label_space)
            print('Number of labels after dataset preconditioning/prefiltering {}'.format(n_labels))
            if n_labels == len(self.label_space):
                break

    def save_xml_data(self):
        self.write_label_names()
        self.write_features()
        self.write_labels()
        self.write_inv_prop()

    def update_propensities(self, A, B):
        print('Calculating inverse propensities... A={}, B={}'.format(A, B))
        self.inv_propensities(A=A, B=B)
        print('Saving to XML txt format...')
        self.write_inv_prop()

    def generate_label_space(self, names, k=512):
        chunks = n_chunks(names, k)

        c = collections.Counter()
        results = Parallel(n_jobs=self.config.analysis.THREAD_POOL_THREADS, verbose=1, backend='multiprocessing')(
                map(delayed(PfastreXML.count_tokens), chunks))
        self.canonical_sets = []
        for s_canonical_set, s_counter in results:
            c += s_counter
            self.canonical_sets += s_canonical_set

        c_tok_k, c_tok_v = zip(*c.most_common(k))
        res = list(c_tok_k)
        return res

    def inv_propensities(self, A=3.0, B=0.5):
        """
            calculate inverse propensity scores P(y_t = 1 | y^*_t = 1) with hyperparameters A and B
        """
        dataset_size, label_space_dims = self.trn_Y.shape
        C = (np.log(dataset_size) - 1) * np.power((B + 1), A)

        # self.logger.info('Labels propensities for '+str(N)+' and '+str(L))
        self.inv_props = np.zeros((label_space_dims,), dtype=float)
        for t in range(label_space_dims):
            n_t = self.trn_Y[t].sum()  # number of data points annotated with label l
            exp_t = np.exp(-A * np.log(n_t + B))
            i_pt = 1.0 + (C * exp_t)
            self.inv_props[t] = i_pt


    @staticmethod
    def write_xml_txt(df, fname, shape=None):
        with open(fname, 'w') as f:
            vshape = df.shape
            if shape:
                vshape = shape

            f.write('{} {}\n'.format(*vshape))
            for index, row in tqdm.tqdm(df.iterrows(), desc='Saving {}'.format(fname), total=vshape[0]):
                rd = row.to_numpy()
                # Extract data, different formats/shapes
                if rd.shape == (1,):
                    rd = rd[0].reshape(-1)

                nz = rd.nonzero()
                I = nz[0]
                first = True
                for i in I:
                    if not first:
                        f.write(' ')
                    # print('Writing row')
                    f.write('{}:{:.6f}'.format(i, rd[i]))
                    first = False
                f.write('\n')

    def read_xml_txt(self, fname):
        with open(fname, 'r') as f:
            first_line = next(f)
            size = first_line.split()
            rows, cols = int(size[0]), int(size[1])
            m = np.zeros((rows, cols), dtype=float)
            for i, row in tqdm.tqdm(enumerate(f), desc='Loading {}'.format(fname), total=rows):
                elems = row.split()
                for elem in elems:
                    ind, val = elem.split(':')
                    m[i, int(ind)] = float(val)
        return m

    def write_labels(self, trn_fname='trn_Y.txt', tst_fname='tst_Y.txt', val_fname='val_Y.txt'):
        PfastreXML.write_xml_txt(self.trn_Y, self.directory + '/' + trn_fname)
        PfastreXML.write_xml_txt(self.tst_Y, self.directory + '/' + tst_fname)
        PfastreXML.write_xml_txt(self.val_Y, self.directory + '/' + val_fname)

    def write_features(self, trn_fname='trn_X.txt', tst_fname='tst_X.txt', val_fname='val_X.txt'):
        ind, row = next(self.X.iterrows())
        embed_len = row['embedding'].shape[-1]
        PfastreXML.write_xml_txt(self.trn_X, self.directory + '/' + trn_fname, shape=(len(self.trn_X), embed_len))
        PfastreXML.write_xml_txt(self.tst_X, self.directory + '/' + tst_fname, shape=(len(self.tst_X), embed_len))
        PfastreXML.write_xml_txt(self.val_X, self.directory + '/' + val_fname, shape=(len(self.val_X), embed_len))

    def write_inv_prop(self, fname='inv_prop.txt'):
        with open(self.directory + '/' + fname, 'w') as f:
            for p in self.inv_props:
                f.write('{:.4f}\n'.format(p))

    def write_label_names(self, fname='labels.txt'):
        with open(self.directory + '/' + fname, 'w') as f:
            for i, l in enumerate(self.label_space):
                f.write('{}:{}\n'.format(i, l))

    def predict_labels(self, score_m_fname='xml_score.mat', threshold=0.1):
        print('Predicting labels against TEST set')
        m = self.read_xml_txt(self.directory + '/' + score_m_fname)
        ml_Y = np.zeros(m.shape)
        for i, row in enumerate(m):
            above_thresh = np.where(row > threshold)
            ranked = np.argsort(row)[::-1]
            ranked_ind_above_threshold = ranked[:len(above_thresh[0])]
            for p in ranked_ind_above_threshold:
                ml_Y[i, p] = 1

            predicted_labels = list(map(lambda x: self.label_space[x], ranked_ind_above_threshold))
            predicted_name = '_'.join(predicted_labels)

            func_id = self.tst_X.index[i]
            func_name = self.name_df['name'][func_id]
            func_real_name = self.real_name_df['real_name'][func_id]
            func_canonl_name = self.canon_name_df['canon_name'][func_id]
            fmt = ' {} {} {} {} => {}'.format(func_id, func_real_name, func_name, func_canonl_name, predicted_name)
            print(fmt)
        return ml_Y

    def ml_scores(self, threshold=0.215, calc_knowns=True, remove_calc=False):
        eval = Evaluation()
        pred_Y = self.predict_labels(threshold=threshold)
        true_Y = self.tst_Y.values

        if calc_knowns:
            # assume we know calculable knowns
            for i, (ind, row) in enumerate(self.tst_Y.iterrows()):
                true_name = self.name_df['name'][ind]
                if true_name in self.calculable_knowns:
                    pred_Y[i, :] = true_Y[i, :]

        if remove_calc:
            # delete calculable knowns
            delete_row_inds = []
            for i, (ind,row) in enumerate(self.tst_Y.iterrows()):
                true_name = self.name_df['name'][ind]
                if true_name in self.calculable_knowns:
                    delete_row_inds.append(i)

        # if args.db_field_name:
            # self.save_predictions(true_Y, pred_Y)

        ml_p = eval.ml_precision(true_Y, pred_Y)
        ml_r = eval.ml_recall(true_Y, pred_Y)
        ml_f1 = eval.ml_f1(ml_p, ml_r)

        # print('Micro Avgs :: Precision:{}, Recall:{} F1:{}'.format(ml_p, ml_r, ml_f1))

        mac_ml_p = eval.ml_precision(true_Y, pred_Y, MODE='MACRO')
        mac_ml_r = eval.ml_recall(true_Y, pred_Y, MODE='MACRO')
        mac_ml_f1 = eval.ml_f1(mac_ml_p, mac_ml_r)

        m_ml_p = mac_ml_p[np.isfinite(mac_ml_p)]
        m_ml_r = mac_ml_r[np.isfinite(mac_ml_r)]
        m_ml_f1 = mac_ml_f1[np.isfinite(mac_ml_f1)]

        macro_avg_ml_p = np.mean(m_ml_p)
        macro_avg_ml_r = np.mean(m_ml_r)
        macro_avg_ml_f1 = np.mean(m_ml_f1)

        experiment_results = {
                'micro_p': ml_p,
                'micro_r': ml_r,
                'micro_f1': ml_f1,
                'macro_p': macro_avg_ml_p,
                'macro_r': macro_avg_ml_r,
                'macro_f1': macro_avg_ml_f1,
                }
        return experiment_results

    # Arguments' index must be aligned with self.tst_Y
    def save_predictions(self, true_Y, pred_Y):
        db = IdaDB()
        db.add_table_column('bcsd_dataset_stripped', f'xfl_true_labels_{args.db_field_name}', 'VARCHAR')
        db.add_table_column('bcsd_dataset_stripped', f'xfl_pred_labels_{args.db_field_name}', 'VARCHAR')
        for i, func_id in enumerate(self.tst_Y.index):
            true_label_indices = list(true_Y[i].nonzero()[0])
            pred_label_indices = list(pred_Y[i].nonzero()[0])
            db_values = {
                    f'xfl_true_labels_{args.db_field_name}': '_'.join(self.label_space[j] for j in true_label_indices),
                    f'xfl_pred_labels_{args.db_field_name}': '_'.join(self.label_space[j] for j in pred_label_indices),
                    }
            db._update_entry('bcsd_dataset_stripped', 'func_id_first', func_id, db_values)

    @staticmethod
    def opt_predict_labels(m, eval, tst_Y, threshold):
        ml_Y = np.zeros(m.shape)
        for i, row in enumerate(m):
            above_thresh = np.where(row > threshold)
            ranked = np.argsort(row)[::-1]
            ranked_ind_above_threshold = ranked[:len(above_thresh[0])]
            for p in ranked_ind_above_threshold:
                ml_Y[i, p] = 1

        pred_Y = ml_Y
        true_Y = tst_Y.values
        ml_p = eval.ml_precision(true_Y, pred_Y)
        ml_r = eval.ml_recall(true_Y, pred_Y)
        ml_f1 = eval.ml_f1(ml_p, ml_r)
        return threshold, ml_f1

    def opt_f1(self, split='val', processes=20):
        print('Ranking labels against ' + split.upper() + ' set')
        m = self.read_xml_txt(self.directory + '/xml_score.mat')
        eval = Evaluation()
        sY = self.val_Y
        if split == 'tst':
            sY = self.tst_Y
        elif split == 'trn':
            sY = self.trn_Y
        with Pool(processes=processes) as p:
            results = p.map(functools.partial(PfastreXML.opt_predict_labels, m, eval, sY), np.linspace(0.001, 0.5, num=500))
        X, Y = zip(*results)
        Y = np.nan_to_num(Y)
        X, Y = list(X), list(Y)
        maxY = Y[np.argmax(Y)]
        optT = X[np.argmax(Y)]
        print(str(X))
        print(str(Y))
        print('Optimal threshold: ' + str(optT) + ' -> ' + str(maxY))
        return optT, maxY

    def evaluate(self, score_m_fname='xml_score.mat', calc_knowns=True):
        m = self.read_xml_txt(self.directory + '/' + score_m_fname)
        xml_eval = Evaluation()
        cgs, dcgs, ndcgs = [], [], []
        np_tst_Y = self.tst_Y.to_numpy()
        p = 5
        for i, row in enumerate(m):
            true_labels = np_tst_Y[i, :]
            our_predict = row
            func_index = self.tst_X.index[i]
            func_name = self.name_df['name'][func_index]
            if calc_knowns:
                if func_name in self.calculable_knowns:
                    our_predict = true_labels
                    np_tst_Y[i, :] = true_labels
            cg = xml_eval.cumulative_gain(true_labels, our_predict, p=p)
            dcg = xml_eval.discounted_cumulative_gain(true_labels, our_predict, p=p)
            ndcg = xml_eval.normalised_discounted_cumulative_gain(true_labels, our_predict, p=p)
            pred_top_n = np.argsort(our_predict)[::-1][:p]
            corr_top_n = np.argsort(true_labels)[::-1][:p]
            pred_labels = tuple(map(lambda x, L=self.label_space: L[x], pred_top_n))
            corr_labels = tuple(map(lambda x, L=self.label_space: L[x], corr_top_n))
            print(
                    '{:<40} :: N={}, TL={}, PL={}, Cumulative Gain: {:>2}, Discounted Cumulative Gain: {:>7}, Normalised Discounted Cumulative Gain: {:>7}'.format(
                        func_name, p, corr_labels, pred_labels, cg, dcg, ndcg
                        ))
            print('\t{:<40}->{:<40}'.format(func_name, '_'.join(pred_labels)))
            cgs.append(cg)
            dcgs.append(dcg)
            ndcgs.append(ndcg)
        acg = np.mean(cgs)
        adcg = np.mean(dcgs)
        andcg = np.mean(ndcgs)

        # Geometric mean
        T = 'N={} :: Mean CG: {:>7}, Mean DCG: {:>7}, Mean NDCG: {:>7}'.format(p, acg, adcg, andcg)
        pr = xml_eval.precision_at_ks(np_tst_Y, m)
        T += ' Precision @n: ' + str(pr)
        print(T)

    @staticmethod
    def opt_score_func(x, self, sg_y):
        A, B = x
        self.inv_propensities(A=A, B=B)
        y = np.array(sorted(list(map(lambda a: 1.0 / a, self.inv_props))))
        g = np.linspace(1, len(y), 26)
        yy = [y[int(x) - 1] for x in g]
        r = np.linalg.norm(sg_y - yy)
        return r

    # Left this function's semantics unchanged from the original XFL code in consultation with Tristan Benoit
    def optimize_hyperparameters(self):
        """
            Uses scipy.optimize.minimize to minimize the distance between propensities of a semi log plot to the 
            sigmoid on a semi-log plot
        """
        sigmoid_f = lambda x, exp=math.exp: 1.0 / (1.0 + exp(-x))
        sg_x = np.linspace(-5, 5, 26)
        sg_y = np.array(list(map(lambda a, f=sigmoid_f: f(a), sg_x)))
        x0 = np.array([0.4, 0.5])
        bnds = ((0, 100), (0, 100))
        res = scipy.optimize.minimize(PfastreXML.opt_score_func, x0, args=(self, sg_y), bounds=bnds)
        print(str(res))
        return res.x[0], res.x[1]

    def precondition_dataset(self, X: pd.DataFrame, Y: pd.DataFrame, L: list):
        """
            Apply preconditions to dataset.

                i)  Remove labels with only a single data point (impossible to learn and test)
                ii) Remove data points with no labels

            WARNING: Will modify label set
        """
        # Only select colums where we have at least min_samples samples
        min_samples = 3
        c1 = Y.sum(axis=0) > min_samples
        c2 = np.where(c1)[0]
        Y = Y[c2]
        l = np.array(L)
        L = l[c2].tolist()

        # Now trim rows to have at least 1 label
        Y = Y.loc[Y.sum(axis=1) > 0]  # axis=1 is rows, only select rows where we have at least 1 label
        X = X.loc[Y.index]  # select rows in X that correspond to the new Y
        print('We now have ' + str(len(X)) + ' rows of data')
        return X, Y, L


# copied from original utils file
def n_chunks(l, n):
    """Break l into n parts"""
    N = math.ceil(len(l) / n)
    for i in range(0, len(l), N):
        yield l[i:i + N]


def ds_to_df(ds, split, split_name, nlp):
    rows = []
    for ex in ds:
        func_id = ex["unique_id"]
        real_name = ex["function_name"]
        name = nlp.strip_library_decorations(real_name)
        canon = nlp.canonical_name(name)
        emb = np.asarray(ex[f"{split}_embedding"], dtype=np.float16)
        rows.append([func_id, real_name, name, canon, emb, split_name])
    df = pd.DataFrame(rows, columns=["func_id","real_name","name","canon_name","embedding","data_split"])
    return df.set_index("func_id")


def load_embedding(teacher_type="clap", data_dir="/mnt/ambrym2/datasets/distil", split="project"):
    config = Config()
    nlp = NLP(config)

    cache_dir = os.path.join(data_dir, ".cache", teacher_type)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "dataset_filter"), exist_ok=True)

    train_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{split}_train.arrow")
    val_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{split}_val.arrow")
    test_cache_filter_path = os.path.join(cache_dir, "dataset_filter", f"{split}_test.arrow")

    ### load dataset
    dataset = load_from_disk(os.path.join(data_dir, f'assembly_x64_1024_{teacher_type}'))
    dataset = dataset.select_columns(["unique_id", "function_name",
        f"{split}_embedding"])
    with open(os.path.join(data_dir, f"cross_{split}_split.json")) as f:
        indices = json.load(f)

    train_ids = set(indices["train"])
    val_ids = set(indices["val"])
    test_ids = set(indices['test'])

    train_dataset = dataset.filter(lambda batch: [uid in train_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=train_cache_filter_path).with_format("python")
    train_df = ds_to_df(train_dataset, split, "train", nlp)
    del train_dataset
    gc.collect()
    val_dataset = dataset.filter(lambda batch: [uid in val_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=val_cache_filter_path).with_format("python")
    val_df = ds_to_df(val_dataset, split, "train_val", nlp)
    del val_dataset
    gc.collect()
    test_dataset = dataset.filter(lambda batch: [uid in test_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=test_cache_filter_path).with_format("python")
    test_df = ds_to_df(test_dataset, split, "test", nlp)
    del test_dataset
    gc.collect()

    df = pd.concat([train_df, val_df, test_df], copy=False)
    return df

def write_results_to_csv(output_file: str, eval_config: dict, metric_results: dict):
    file_path = Path(output_file)
    file_exists = file_path.is_file()
    with file_path.open('a+', newline='') as file:
        csv_entry = eval_config | metric_results
        csv_writer = csv.DictWriter(file, fieldnames=csv_entry.keys())
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerow(csv_entry)


def main(args, temp_dir):
    print(f'{args=}')
    print('Loading embedding and making dataframe...')
    df = load_embedding(teacher_type=f"{args.teacher_type}", split=args.split)
    xml = PfastreXML(Config(), temp_dir)
    xml.from_data(df, args.labelspace_dims)

    print('Optimizing A and B from PfastreXML...')
    a, b = xml.optimize_hyperparameters()
    xml.update_propensities(a, b)

    print('Training PfastreXML...')
    PfastreXML.train_pfastrexml(args.pfastrexml_train_path, xml.directory, 'trn_X.txt', 'trn_Y.txt', trees=args.trees,
                                a=args.a, g=args.g)

    print('Optimizing ML threshold...')
    PfastreXML.pred_pfastrexml(args.pfastrexml_predict_path, xml.directory, 'val_X.txt')
    threshold, opt_f1 = xml.opt_f1(split='val')
    if np.isnan(opt_f1):
        print('No data on validation split for this embedding, using training split to select threshold')
        PfastreXML.pred_pfastrexml(args.pfastrexml_predict_path, xml.directory, 'trn_X.txt')
        threshold, _ = xml.opt_f1(split='trn')

    print('Computing results...')
    PfastreXML.pred_pfastrexml(args.pfastrexml_predict_path, xml.directory, 'tst_X.txt')
    xml.evaluate(calc_knowns=True)
    ml_scores = xml.ml_scores(threshold=threshold, calc_knowns=True, remove_calc=False)
    eval_config = {
        'teacher_type': args.teacher_type,
        'split': args.split,
        'labelspace_dims': args.labelspace_dims,
        'trees': args.trees,
        'a': args.a,
        'g': args.g,
    }
    write_results_to_csv(f"{args.output_dir}/results_xfl_{args.teacher_type}-{args.split}", eval_config, ml_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_type', type=str, default="clap")
    parser.add_argument('--split', type=str, default="project")
    parser.add_argument('--pfastrexml_train_path', default='/mnt/ambrym2/datasets/distil/xfl/PfastreXML_train')
    parser.add_argument('--pfastrexml_predict_path', default='/mnt/ambrym2/datasets/distil/xfl/PfastreXML_predict')
    parser.add_argument('-l', '--labelspace-dims', default=1024, type=int, help='Size of labelspace to use')
    parser.add_argument('-trees', '--trees', default=256, type=int, help='Number of trees of pfastreXML')
    parser.add_argument('-a', '--a', default=1, type=float, help='Hyperparameter a of pfastreXML')
    parser.add_argument('-g', '--g', default=30, type=int, help='Hyperparameter g of pfastreXML')
    parser.add_argument('--db_field_name', type=str)  # will not save to database if not set
    parser.add_argument('--output_dir', type=str, default='/mnt/ambrym2/datasets/distil/xfl')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix='xfl-eval-') as temp_dir:
        main(args, temp_dir)
