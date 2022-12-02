import pandas as pd
import numpy as np
from IPython.display import display
from math import log2
from pprint import pp


qrels = pd.read_csv('qrels.csv')
results = pd.read_csv('system_results.csv')


display(qrels)
display(results)

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

def generate_eval_dict(num_systems, num_queries, qrels_df, results_df):
    eval_dict = {}
    for q in range(1, num_queries+1):
        eval_dict.update({q : {} })
        ideal_dcg_10 = get_ideal_dcg(q, 10, qrels_df)
        ideal_dcg_20 = get_ideal_dcg(q, 20, qrels_df)
        for s in range(1, num_systems+1):
            norm_10 = norm_dcg(q, ideal_dcg_10, s, 10, qrels_df, results_df)
            norm_20 = norm_dcg(q, ideal_dcg_20, s, 20, qrels_df, results_df)
            eval_dict.get(q).update({s : {'nDCG@10' : norm_10, 'nDCG@20' : norm_20}})
    return eval_dict


def confusion_matrix(qrels_df, results_df, k, query, system):
    query_docs_ids = qrels_df[qrels_df['query_id'] == query]['doc_id']
    print(query_docs_ids.values)
    
    

num_systems = len(results['system_number'].unique())
num_queries = len(qrels['query_id'].unique())

#eval_dict = generate_eval_dict(num_systems, num_queries, qrels, results)
#pp(eval_dict)

confusion_matrix(qrels, results, 10, 1, 1)
