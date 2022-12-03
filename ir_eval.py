import pandas as pd
import numpy as np
from math import log2
from pprint import pp
from scipy import stats
metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']


def get_relevance_rank_actual(query, system, k, qrels_df, results_df):
    rels_for_q = qrels_df[qrels_df['query_id'] == query]
    output_for_q = results_df[(results_df['query_number'] == query) & (results_df['system_number'] == system)]

    # print('query {}\nsystem {}'.format(query, system))
    # print('relevant_for_query\n' + str(rels_for_q))
    # print('system_output\n' + str(output_for_q))

    relevance_rank = []
    for i in range(k):
        result_doc = output_for_q.iloc[i]
        relevant_doc = rels_for_q[rels_for_q['doc_id'] == result_doc['doc_number']]  

        #relevant doc may be empty
        if relevant_doc.shape[0] > 0:
            relevance = relevant_doc.iloc[0]['relevance']
            doc_rank = result_doc['rank_of_doc']
            relevance_rank.append((relevance, doc_rank))
    relevance_rank.sort(key=lambda x: x[1])
    return relevance_rank

# is the same for every system
def get_relevance_rank_ideal(query, k, qrels_df):
    relevance_rank = []
    rels_for_q = qrels_df[qrels_df['query_id'] == query]
    
    num_rels = rels_for_q.shape[0]
    # ensures we don't try to index df when k > num_rels, avoid index oob
    if num_rels < k:
        k = num_rels
    for i in range(k):
        relevant_doc = rels_for_q.iloc[i]
        rank = i+1 # rank starts at 1, index starts at 0
        relevance = relevant_doc['relevance']
        relevance_rank.append((relevance, rank))
    return relevance_rank 
        

def calculate_dcg_for_rr(relevance_rank):
    dcg = 0
    if relevance_rank:
        dcg = relevance_rank[0][0]
    for i in range(1,len(relevance_rank)):
        # where a doc has relevance 0 the corresponding term would be 0 
        # so relevance rank just ignores it
        next_term = (relevance_rank[i][0]/log2(relevance_rank[i][1]))
        dcg += next_term
    return dcg

def get_ideal_dcg(query, k, qrels_df):
    relevance_rank_ideal = get_relevance_rank_ideal(query, k, qrels_df)
    ideal_dcg = calculate_dcg_for_rr(relevance_rank_ideal)
    return ideal_dcg
    

def get_dcg(query, system, k, qrels_df, results_df):
    relevance_rank_actual = get_relevance_rank_actual(query, system, k, qrels_df, results_df)
    dcg_at_k = calculate_dcg_for_rr(relevance_rank_actual)
    return dcg_at_k
    
def norm_dcg(query, ideal_dcg, system, k, qrels_df, results_df):
    dcg = get_dcg(query, system, k, qrels_df, results_df)
    norm_dcg = dcg/ideal_dcg
    return norm_dcg


def confusion_matrix(qrels_df, results_df, k, query, system):
    rel_ids= qrels_df[qrels_df['query_id'] == query]['doc_id'].values[:k]
    results = results_df[(results_df['query_number'] == query) & (results_df['system_number'] == system)]['doc_number'].values[:k]
    tp = 0
    fp = 0
    fn = 0
    for result in results:
        if result in rel_ids:
            tp += 1
        else: 
            fp += 1
    
    for rel in rel_ids:
        if rel not in results:
            fn += 1
    return tp, fp, fn

def precision(qrels_df, results_df, k, query, system):
    tp, fp, fn = confusion_matrix(qrels_df, results_df, k, query, system)
    precision = tp/(tp + fp)
    return precision

def r_precision(qrels_df, results_df, query, system):
    num_relevant = qrels_df[qrels_df['query_id'] == query].shape[0]
    tp, fp, fn = confusion_matrix(qrels_df, results_df, num_relevant, query, system)
    precision = tp/(tp + fp)
    return precision

def recall(qrels_df, results_df, k, query, system):
    tp, fp, fn = confusion_matrix(qrels_df, results_df, k, query, system)
    recall = tp/(tp+fn)
    return recall

def average_precision(qrels_df, results_df, query, system):
    relevant_docs = qrels_df[qrels_df['query_id'] == query]['doc_id'].values
    retrieved_docs = results_df[(results_df['system_number'] == system) & (results_df['query_number'] == query)]['doc_number'].values
    average_precision = 0
    for k,retrieved in enumerate(retrieved_docs,1):
        if retrieved in relevant_docs:
            average_precision += precision(qrels_df, results_df, k, query, system)
    average_precision = average_precision/len(relevant_docs)
    return average_precision 


def generate_eval_dict(num_systems, num_queries, qrels_df, results_df):
    eval_dict = {}
    for q in range(1, num_queries+1):
        eval_dict.update({q : {} })
        ideal_dcg_10 = get_ideal_dcg(q, 10, qrels_df)
        ideal_dcg_20 = get_ideal_dcg(q, 20, qrels_df)
        for s in range(1, num_systems+1):
            ave_precision = average_precision(qrels_df, results_df, q, s)
            precision_10 = precision(qrels_df, results_df, 10, q, s)
            recall_50 = recall(qrels_df, results_df, 50, q, s)
            r_prec = r_precision(qrels_df, results_df, q, s)
            norm_10 = norm_dcg(q, ideal_dcg_10, s, 10, qrels_df, results_df)
            norm_20 = norm_dcg(q, ideal_dcg_20, s, 20, qrels_df, results_df)
            eval_dict.get(q).update({s : {'nDCG@10' : norm_10, 'nDCG@20' : norm_20,
             'P@10' : precision_10, 'R@50' : recall_50, 'r-precision' : r_prec, 'AP' : ave_precision}})
    
   
    return eval_dict

def generate_mean_dict(eval_dict, num_systems, num_queries):
    metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
    mean_dict = {}
    sdev_dict = {}
    for s in range(1,num_systems+1):
        mean_dict.update({s : {}})
        sdev_dict.update({s : {}})
        for q in range(1,num_queries+1):
            for metric in metrics:
                if not mean_dict.get(s).get(metric):
                    mean_dict.get(s).update({metric : [eval_dict.get(q).get(s).get(metric)]})
                else:
                    mean_dict.get(s)[metric].append(eval_dict.get(q).get(s).get(metric))
        for metric in metrics:
            scores = mean_dict.get(s)[metric]
            mean_dict.get(s)[metric] = np.mean(scores)
            sdev_dict.get(s).update({metric : np.std(scores)})
    return mean_dict, sdev_dict


            
def write_eval_to_file(eval_dict, mean_dict, num_systems, num_queries):
    metrics = ['P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20']
    f = open('ir_eval.csv', 'w')
    f.write('system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n')
    for s in range(1,num_systems+1):
        for q in range(1,num_queries+1):
            f.write('{},{}'.format(s, q))
            for metric in metrics:
                f.write(',{:.3f}'.format(eval_dict.get(q).get(s).get(metric)))
            f.write('\n')
            if q == num_queries:
                f.write('{},{}'.format(s, 'mean'))
                for metric in metrics:
                    f.write(',{:.3f}'.format(mean_dict.get(s).get(metric)))
                if s != num_systems:
                    f.write('\n')
    f.close()

        
def EVAL(qrel_filename, system_results_filename):  
    qrels = pd.read_csv(qrel_filename)
    results = pd.read_csv(system_results_filename)
    num_queries = len(qrels['query_id'].unique())
    num_systems = len(results['system_number'].unique())
    eval_dict = generate_eval_dict(num_systems, num_queries, qrels, results)
    mean_dict, sdev_dict = generate_mean_dict(eval_dict, num_systems, num_queries)
    write_eval_to_file(eval_dict, mean_dict, num_systems, num_queries)
    return eval_dict, sdev_dict, mean_dict, num_systems, num_queries

def get_top_two_dict(mean_dict, num_systems):
    top_two_dict = {}
    for metric in metrics:
        metric_means = []
        for s in range(1,num_systems+1):
            metric_means.append((s, mean_dict.get(s).get(metric)))
        metric_means.sort(key=lambda x: x[1], reverse=True)
        top_two_dict.update({metric : metric_means[:2]})
    return top_two_dict

def is_significant(mean_dict, sdev_dict, system1, system2, alpha, metric):
    zscore = (mean_dict.get(system2).get(metric) - mean_dict.get(system1).get(metric)) / sdev_dict.get(system1).get(metric)
    p_val = stats.norm.cdf(zscore)
    # print(mean_dict.get(system2).get(metric))
    # print(mean_dict.get(system1).get(metric))
    # print(p_val)
    sig = False
    if p_val < alpha:
        sig = True
    return sig

def significance_for_all(mean_dict, sdev_dict, num_systems):
    top_two_dict = get_top_two_dict(mean_dict, num_systems)
    for metric in metrics:
        top_two = top_two_dict.get(metric)
        top_system = top_two[0][0]
        second_system = top_two[1][0]
        sig = is_significant(mean_dict, sdev_dict, top_system, second_system, 0.0025, metric)
        place = ' NOT '
        if sig:
            place = ' '
            
        print('top system for {met}: {sys1}\nsecond system for {met}: {sys2}\n'
              'top system is{place}significantly better than second\n\n'.format(met=metric, sys1=top_system, sys2=second_system, place=place))

eval_dict, sdev_dict, mean_dict, num_systems, num_queries = EVAL('qrels.csv','system_results.csv')
significance_for_all(mean_dict, sdev_dict, num_systems)