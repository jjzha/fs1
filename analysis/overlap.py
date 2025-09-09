from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json

test_dataset = load_dataset("jjzha/fs1-benchmark-2708", split="test")

def preprocess_fn(example):
    example["question"] = example["question"].lower()
    return example

test_dataset = test_dataset.map(preprocess_fn)
df_test = test_dataset.to_pandas()

grouped_test = df_test.groupby("source_of_data")["question"].apply(list).to_dict()

train_dataset = load_dataset("jjzha/fs1-labeled", split="train")
train_dataset = train_dataset.filter(lambda example: example['label'] == 1).map(preprocess_fn)
df_train = train_dataset.to_pandas()

train_questions = df_train["question"].tolist()
grouped_train = {"CWQ_train": train_questions}

grouped_questions = {**grouped_test, **grouped_train}

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

encoded_sources = {}
for source, questions in grouped_questions.items():
    embeddings = model.encode(questions, convert_to_tensor=True)
    encoded_sources[source] = embeddings

results = {}

for src1, questions1 in grouped_questions.items():
    for src2, questions2 in grouped_questions.items():
        if src1 >= src2:
            continue
        
        emb1 = encoded_sources[src1]
        emb2 = encoded_sources[src2]
        
        cosine_scores = util.cos_sim(emb1, emb2)
        similar_count = (cosine_scores > 0.90).sum().item()
        exact_match_count = sum(1 for q1 in questions1 for q2 in questions2 if q1 == q2)
        average_similarity = cosine_scores.mean().item()
        
        results[f"{src1}${src2}"] = {
            "similar_count": similar_count,
            "exact_match_count": exact_match_count,
            "average_similarity": average_similarity
        }

output_file = "sentence_similarity_counts_2808.json"
with open(output_file, "w") as jsonfile:
    json.dump(results, jsonfile, indent=4)

print(f"Sentence similarity counts have been written to {output_file}\n")

print("Sentence similarity counts between groups:")
for key, counts in results.items():
    print(f"{key}: Similar Count = {counts['similar_count']}, Exact Match Count = {counts['exact_match_count']}, Average Similarity = {counts['average_similarity']:.4f}")
