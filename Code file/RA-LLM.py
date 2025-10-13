# source activate /root/pyenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "/root/autodl-tmp/model/bge-large-zh"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}
emb_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
print(emb_model)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
from ragas import evaluate
from ragas.metrics import answer_similarity, answer_correctness
from datasets import Dataset
import pandas as pd
import json

import os
from langchain_community.llms import BaichuanLLM

os.environ["BAICHUAN_API_KEY"] = "sk-5c08af67360cd0ef9f6841adeb24080c"

# Load the model
llm = BaichuanLLM()
print(llm)




path = "output.xlsx"
df = pd.read_excel(path)

columns_to_merge=["top1", "top2", "top3"]
# questions, answers, contexts, ground_truths = df['query'].tolist(), df['rag_answer'].tolist(), df[columns_to_merge].values.tolist(), df['answer_s'].tolist()
# questions, answers, contexts, ground_truths = df['query'][:5].tolist(), df['rag_answer'][:5].tolist(), df[columns_to_merge].values[:5].tolist(), df['answer_s'][:5].tolist()
# # 填充自己的数据内容
# evalsets = {
#               "question": questions,
#               "answer": answers,
#               "contexts": contexts,
#               "ground_truth": ground_truths
#             }
# evalsets = Dataset.from_dict(evalsets)

# from datasets import load_dataset

# ragas_dataset = load_dataset('json', data_files='data.json')
# data = ragas_dataset['train']
# result = evaluate(
#   data,
#   metrics=[
#       answer_correctness,
#       answer_similarity
#   ],
#   raise_exceptions=False,
#   llm=llm, 
#   embeddings=emb_model
# )

# print(result)

# for i in range(62):
#     questions, answers, contexts, ground_truths = df['query'][i:i+1].tolist(), df['rag_answer'][i:i+1].tolist(), df[columns_to_merge].values[i:i+1].tolist(), df['answer_s'][i:i+1].tolist()
#     # 填充自己的数据内容
#     evalsets = {
#                   "question": questions,
#                   "answer": answers,
#                   "contexts": contexts,
#                   "ground_truth": ground_truths
#                 }
#     data = Dataset.from_dict(evalsets)

#     results = evaluate(data,metrics=[answer_similarity, answer_correctness], llm=llm, embeddings=emb_model)
    
#     answer_similarity.append(results["answer_similarity"])
#     answer_correctness.append(results["answer_correctness"])
    
# df["answer_similarity"]=answer_similarity
# df["answer_correctness"]=answer_correctness
def get_metrics(i):
    questions, answers, contexts, ground_truths = df['query'][i:i+1].tolist(), df['llm_answer'][i:i+1].tolist(), df[columns_to_merge].values[i:i+1].tolist(), df['answer_s'][i:i+1].tolist()
    # 填充自己的数据内容
    evalsets = {
                  "question": questions,
                  "answer": answers,
                  "contexts": contexts,
                  "ground_truth": ground_truths
                }
    data = Dataset.from_dict(evalsets)

    results = evaluate(data,metrics=[answer_similarity, answer_correctness], llm=llm, embeddings=emb_model)
    return results["answer_similarity"], results["answer_correctness"]

answer_similarity_list, answer_correctness_list = [],[]
for i in range(62):
    s,c = get_metrics(i)
    answer_similarity_list.append(s)
    answer_correctness_list.append(c)
    print("Finished index: "+str(i))
    
df["llm_answer_similarity"]=answer_similarity_list
df["llm_nswer_correctness"]=answer_correctness_list

df.to_excel('output.xlsx', index=False)
# with open('res.json','w')as f:
#     json.dump(results,f)