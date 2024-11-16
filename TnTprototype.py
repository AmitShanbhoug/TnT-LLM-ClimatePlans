from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import re
from typing import List, Dict
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb
import numpy as np
import csv

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper to calculate the number of tokens in a prompt
def count_tokens(prompt: str) -> int:
    # Estimate number of tokens by splitting the prompt into words
    return len(prompt.split())

# Function to count tokens in a string (rough estimate)
def count_tokens(text: str) -> int:
    # Assuming each word is a token on average
    return len(text.split())

# Updated function to handle batches and ensure token limit is not exceeded
def call_gpt_api_batch(prompts: List[str], model="gpt-4o-mini", max_tokens=7000) -> List[str]:
    responses = []
    for prompt in prompts:
        token_count = count_tokens(prompt)
        
        # Ensure token count for prompt + completion does not exceed max_tokens
        if token_count > max_tokens:
            print(f"Warning: Prompt is too long ({token_count} tokens). It will not be sent.")
            responses.append("")  # Skip this prompt if it's too long
            continue
        
        try:
            # Make GPT API call (ensure response doesn't exceed token limit)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,  # Set max tokens for the response
                top_p=1,
                n=1,
                stop=None
            )
            responses.append(response.choices[0].message.content.strip())

        except Exception as e:
            print(f"Error calling GPT API: {e}")
            responses.append("")  # Append empty string in case of error
            time.sleep(1)  # Brief pause to avoid rate limits
    return responses

# Preprocess text: remove non-alphabetic characters and extra spaces
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zAZ\s]', '', text)
    text = " ".join(text.split())
    return text

# Read Description from Excel file
def read_Description_from_excel(file_path: str) -> List[str]:
    df = pd.read_excel(file_path)
    return df['Description'].tolist()

# Function to handle batching of documents and calling GPT API
def summarize_documents(descriptions: List[str], use_case: str, batch_size=15) -> List[str]:
    summaries = []
    batched_prompts = []
    
    # Generate prompts for each description
    for desc in descriptions:
        prompt = f"Summarize this Description in about 20 words for the use case: {use_case}\n\n{desc}"
        batched_prompts.append(prompt)
    
    # Process in batches
    for i in range(0, len(batched_prompts), batch_size):
        batch = batched_prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(batched_prompts) + batch_size - 1) // batch_size}")
        batch_summaries = call_gpt_api_batch(batch)
        summaries.extend(batch_summaries)
    
    return summaries

def generate_taxonomy(summaries: List[str], use_case: str, iterations: int = 3, batch_size: int = 5) -> List[Dict]:
    taxonomy = []
    for _ in range(iterations):
        batches = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]
        for batch in batches:
            prompt = (
                f"Based on these summaries, refine and generate a taxonomy for the use case: {use_case}.\n\n"
                f"Summaries:\n{batch}\n\n"
                f"Current Taxonomy: {taxonomy}"
            )
            updated_taxonomy = call_gpt_api_batch([prompt])[0]  # Only one prompt per batch
            taxonomy.append(updated_taxonomy)
    
    review_prompt = f"Review and refine the following taxonomy for the use case: {use_case}\n\n{taxonomy}"
    final_taxonomy = call_gpt_api_batch([review_prompt])[0]
    
    return final_taxonomy

def evaluate_taxonomy(taxonomy: List[Dict], summaries: List[str], use_case: str):
    labels = generate_pseudo_labels(summaries, taxonomy)
    coverage = 1 - (labels.count('Other') + labels.count('Undefined')) / len(labels)
    
    accuracy_prompt = f"Given the use case '{use_case}', which label is more accurate for this text:\nText: {{text}}\nLabel 1: {{label1}}\nLabel 2: {{label2}}"
    accuracy_hits = 0
    for summary in random.sample(summaries, min(100, len(summaries))):
        true_label = generate_pseudo_labels([summary], taxonomy)[0]
        random_label = random.choice([l for l in set(labels) if l != true_label])
        response = call_gpt_api_batch([accuracy_prompt.format(text=summary, label1=true_label, label2=random_label)])[0]
        if "Label 1" in response:
            accuracy_hits += 1
    label_accuracy = accuracy_hits / 100
    
    relevance_prompt = f"Is this label relevant to the use case '{use_case}'?\nLabel: {{label}}"
    relevance_hits = 0
    for label in set(labels):
        response = call_gpt_api_batch([relevance_prompt.format(label=label)])[0]
        if "yes" in response.lower():
            relevance_hits += 1
    relevance = relevance_hits / len(set(labels))
    
    return coverage, label_accuracy, relevance

# Phase 2: LLM-Augmented Text Classification
def generate_pseudo_labels(summaries: List[str], taxonomy: List[Dict]) -> List[str]:
    labels = []
    for summary in summaries:
        prompt = f"Assign the most relevant label from this taxonomy:\n{taxonomy}\n\nText:\n{summary}"
        label = call_gpt_api_batch([prompt])[0]
        labels.append(label)
    return labels

def train_lightweight_classifiers(documents: List[str], labels: List[str]):
    processed_documents = [preprocess_text(doc) for doc in documents]
    processed_documents = [doc for doc in processed_documents if doc.strip()]

    if not processed_documents:
        print("No valid documents for training!")
        return None, None, None

    le = LabelEncoder()
    y = le.fit_transform(labels)

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(processed_documents)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'min_data_in_leaf': 1,
        'min_data_in_bin': 1,
        'verbose': -1
    }
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    lgb_predictions = lgb_model.predict(X_test.toarray())
    lgb_predictions = np.argmax(lgb_predictions, axis=1)
    
    unique_classes = np.unique(y)
    target_names = [le.classes_[i] for i in unique_classes]
    
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions, labels=unique_classes, target_names=target_names))
    print("MLP Classification Report:")
    print(classification_report(y_test, mlp_predictions, labels=unique_classes, target_names=target_names))
    print("LightGBM Classification Report:")
    print(classification_report(y_test, lgb_predictions, labels=unique_classes, target_names=target_names))
    
    return (lr_model, mlp_model, lgb_model), tfidf, le

def save_results(taxonomy, evaluation_metrics, classification_reports):
    with open('generated_taxonomy.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Taxonomy'])
        writer.writerow([taxonomy])

    with open('evaluation_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in evaluation_metrics.items():
            writer.writerow([metric, value])

    for model_name, report in classification_reports.items():
        report_lines = report.split('\n')
        report_data = []
        for line in report_lines[2:-5]:
            row = line.split()
            if len(row) == 5:
                report_data.append(row)
            elif len(row) == 6:
                report_data.append([row[0], row[1], row[2], row[3], row[4]])
        
        df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        df.to_csv(f'{model_name}_classification_report.csv', index=False)

def main(file_path: str, use_case: str):
    Description = read_Description_from_excel(file_path)
    summaries = summarize_documents(Description, use_case)
    taxonomy = generate_taxonomy(summaries, use_case)
    
    coverage, label_accuracy, relevance = evaluate_taxonomy(taxonomy, summaries, use_case)
    evaluation_metrics = {
        "Taxonomy Coverage": coverage,
        "Label Accuracy": label_accuracy,
        "Relevance to Use-case": relevance
    }
    
    pseudo_labels = generate_pseudo_labels(summaries, taxonomy)
    classifiers, tfidf_vectorizer, label_encoder = train_lightweight_classifiers(summaries, pseudo_labels)
    
    if classifiers and tfidf_vectorizer:
        lr_model, mlp_model, lgb_model = classifiers
        X_test = tfidf_vectorizer.transform(summaries)
        y_test = label_encoder.transform(pseudo_labels)
        
        unique_classes = np.unique(y_test)
        target_names = [label_encoder.classes_[i] for i in unique_classes]
        
        classification_reports = {
            "Logistic_Regression": classification_report(y_test, lr_model.predict(X_test), labels=unique_classes, target_names=target_names),
            "MLP": classification_report(y_test, mlp_model.predict(X_test), labels=unique_classes, target_names=target_names),
            "LightGBM": classification_report(y_test, np.argmax(lgb_model.predict(X_test.toarray()), axis=1), labels=unique_classes, target_names=target_names)
        }
        
        save_results(taxonomy, evaluation_metrics, classification_reports)
        
        return classifiers, tfidf_vectorizer, taxonomy, label_encoder
    else:
        print("Error in training the classifiers!")
        return None, None, None, None

if __name__ == "__main__":
    file_path = "climate_adaptation_solutions.xlsx"
    use_case = "identify tech-enabled solutions for climate adaptation and resilience"
    main(file_path, use_case)