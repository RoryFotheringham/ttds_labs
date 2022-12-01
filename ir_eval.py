import pandas as pd
from IPython.display import display


qrels_df = pd.read_csv('qrels.csv')
results_df = pd.read_csv('system_results.csv')
# with open('qrels.csv','r') as f:
#     qrels = f.read()
    
# with open('system_results.csv','r') as f:
#     results = f.read()

# qrels_df = pd.read_csv(qrels)
# results_df = pd.read_csv(results)

display(qrels_df)
display(results_df)

q = 1
system = 1
k = 10
rels_for_q = qrels_df[qrels_df['query_id'] == q]
output_for_q = results_df[(results_df['query_number'] == q) & (results_df['system_number'] == system)]

for i in range(k):
    ranked_doc = output_for_q['doc_number'].iloc(i)
    rel_of_doc = rels_for_q[rels_for_q['doc_id'] == ranked_doc]['relevance'] # first check that the ranked doc is in the series
    
    
print(rels_for_q)
print(output_for_q)

