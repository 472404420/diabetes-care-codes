import pandas as pd
import bert_score
import sacrebleu
from rouge_score import rouge_scorer



# Read the Excel file
df = pd.read_excel('output.xlsx')

columns_to_merge=["top1", "top2", "top3"]

# Extract columns
queries = df['query'].tolist()
answers = df['rag_answer'].tolist()
context = df[columns_to_merge].values.tolist()
ground_truth = df['answer_s'].tolist()

# # Calculate BERTScore
# P, R, F1 = bert_score.score(answers, ground_truth, lang='zh', verbose=True)

# # Convert tensors to lists for easier handling
# P = P.tolist()
# R = R.tolist()
# F1 = F1.tolist()

# # Print BERTScore results
# for i, (p, r, f1) in enumerate(zip(P, R, F1)):
#     print(f'Query {i+1}: Precision: {p}, Recall: {r}, F1: {f1}')

# import sacrebleu

# # Calculate BLEU score
# bleu = sacrebleu.corpus_bleu(answers, [ground_truth])

# # Print BLEU score
# print(f'BLEU score: {bleu.score}')

# from rouge_score import rouge_scorer

# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# # Calculate ROUGE scores
# rouge_scores = [scorer.score(gt, ans) for gt, ans in zip(ground_truth, answers)]

# # Print ROUGE scores
# for i, scores in enumerate(rouge_scores):
#     print(f'Query {i+1}: ROUGE-1: {scores["rouge1"].fmeasure}, ROUGE-2: {scores["rouge2"].fmeasure}, ROUGE-L: {scores["rougeL"].fmeasure}')

P_list, R_list, F1_list, rouge_l_list = [],[],[],[]

# Calculate BERTScore
P, R, F1 = bert_score.score(answers, ground_truth, lang='zh', verbose=True)
P, R, F1 = P.tolist(), R.tolist(), F1.tolist()

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(answers, [ground_truth])

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = [scorer.score(gt, ans) for gt, ans in zip(ground_truth, answers)]

answer_similarity_list, answer_correctness_list = [],[]

# Print results
for i, (p, r, f1, rouge) in enumerate(zip(P, R, F1, rouge_scores)):
    P_list.append(p)
    R_list.append(r)
    F1_list.append(f1)
    rouge_l_list.append(rouge)

print(f'Overall BLEU score: {bleu.score}')

# df["llm_bertscore_p"]=P_list
# df["llm_bertscore_r"]=R_list
# df["llm_bertscore_f1"]=F1_list
# df["llm_rouge-L"]=rouge_l_list

# df.to_excel('output.xlsx', index=False)