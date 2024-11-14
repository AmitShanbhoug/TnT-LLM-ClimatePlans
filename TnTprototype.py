from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import re
from typing import List, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb
import fitz 
import numpy as np
import csv

load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Call GPT API (using the updated v1.0.0+ method)
def call_gpt_api(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            n=1,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return ""

# Preprocess text: remove non-alphabetic characters and extra spaces
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join(text.split())
    return text

# Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

# Read documents from a folder
def read_documents_from_folder(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            if text:
                documents.append(text)
    return documents

# Phase 1: Taxonomy Generation

def summarize_documents(documents: List[str], use_case: str) -> List[str]:
    summaries = []
    for doc in documents:
        prompt = f"Summarize this document in about 20 words for the use case: {use_case}\n\n{doc[:2000]}"
        summary = call_gpt_api(prompt)
        summaries.append(summary)
    return summaries

def generate_taxonomy(summaries: List[str], use_case: str, iterations: int = 3, batch_size: int = 200) -> List[Dict]:
    taxonomy = []
    for _ in range(iterations):
        batches = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]
        for batch in batches:
            prompt = (
                f"Based on these summaries, refine and generate a taxonomy for the use case: {use_case}.\n\n"
                f"Summaries:\n{batch}\n\n"
                f"Current Taxonomy: {taxonomy}"
            )
            updated_taxonomy = call_gpt_api(prompt)
            taxonomy.append(updated_taxonomy)
    
    # Final review prompt
    review_prompt = f"Review and refine the following taxonomy for the use case: {use_case}\n\n{taxonomy}"
    final_taxonomy = call_gpt_api(review_prompt)
    
    return final_taxonomy

def evaluate_taxonomy(taxonomy: List[Dict], summaries: List[str], use_case: str):
    # Taxonomy Coverage
    labels = generate_pseudo_labels(summaries, taxonomy)
    coverage = 1 - (labels.count('Other') + labels.count('Undefined')) / len(labels)
    
    # Label Accuracy
    accuracy_prompt = f"Given the use case '{use_case}', which label is more accurate for this text:\nText: {{text}}\nLabel 1: {{label1}}\nLabel 2: {{label2}}"
    accuracy_hits = 0
    for summary in random.sample(summaries, min(100, len(summaries))):
        true_label = generate_pseudo_labels([summary], taxonomy)[0]
        random_label = random.choice([l for l in set(labels) if l != true_label])
        response = call_gpt_api(accuracy_prompt.format(text=summary, label1=true_label, label2=random_label))
        if "Label 1" in response:
            accuracy_hits += 1
    label_accuracy = accuracy_hits / 100
    
    # Relevance to Use-case Instruction
    relevance_prompt = f"Is this label relevant to the use case '{use_case}'?\nLabel: {{label}}"
    relevance_hits = 0
    for label in set(labels):
        response = call_gpt_api(relevance_prompt.format(label=label))
        if "yes" in response.lower():
            relevance_hits += 1
    relevance = relevance_hits / len(set(labels))
    
    return coverage, label_accuracy, relevance

# Phase 2: LLM-Augmented Text Classification

def generate_pseudo_labels(summaries: List[str], taxonomy: List[Dict]) -> List[str]:
    labels = []
    for summary in summaries:
        prompt = f"Assign the most relevant label from this taxonomy:\n{taxonomy}\n\nText:\n{summary}"
        label = call_gpt_api(prompt)
        labels.append(label)
    return labels

def train_lightweight_classifiers(documents: List[str], labels: List[str]):
    processed_documents = [preprocess_text(doc) for doc in documents]
    processed_documents = [doc for doc in processed_documents if doc.strip()]

    if not processed_documents:
        print("No valid documents for training!")
        return None, None

    # Convert labels to numerical format
    le = LabelEncoder()
    y = le.fit_transform(labels)

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(processed_documents)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    
    # MLP Classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    
    # LightGBM
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
    
    # Determine unique classes
    unique_classes = np.unique(y)
    
    # Print classification reports for each model, using only unique classes
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions, 
                                labels=unique_classes, 
                                target_names=[le.classes_[i] for i in unique_classes]))

    print("MLP Classification Report:")
    print(classification_report(y_test, mlp_predictions, 
                                labels=unique_classes, 
                                target_names=[le.classes_[i] for i in unique_classes]))

    print("LightGBM Classification Report:")
    print(classification_report(y_test, lgb_predictions, 
                                labels=unique_classes, 
                                target_names=[le.classes_[i] for i in unique_classes]))
    
    return (lr_model, mlp_model, lgb_model), tfidf, le

def save_results(taxonomy, evaluation_metrics, classification_reports):
    # Save taxonomy
    with open('generated_taxonomy.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Taxonomy'])
        writer.writerow([taxonomy])

    # Save evaluation metrics
    with open('evaluation_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric, value in evaluation_metrics.items():
            writer.writerow([metric, value])

    # Save classification reports
    for model_name, report in classification_reports.items():
        # Convert the classification report to a DataFrame
        report_lines = report.split('\n')
        report_data = []
        for line in report_lines[2:-5]:  # Skip header and footer
            row = line.split()
            if len(row) == 5:  # It's a regular row
                report_data.append(row)
            elif len(row) == 6:  # It's the 'accuracy' row
                report_data.append([row[0], row[1], row[2], row[3], row[4]])
        
        df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        df.to_csv(f'{model_name}_classification_report.csv', index=False)

def main(folder_path: str, use_case: str):
    # Phase 1: Taxonomy Generation
    documents = read_documents_from_folder(folder_path)
    summaries = summarize_documents(documents, use_case)
    taxonomy = generate_taxonomy(summaries, use_case)
    
    coverage, label_accuracy, relevance = evaluate_taxonomy(taxonomy, summaries, use_case)
    evaluation_metrics = {
        "Taxonomy Coverage": coverage,
        "Label Accuracy": label_accuracy,
        "Relevance to Use-case": relevance
    }
    
    # Phase 2: LLM-Augmented Text Classification
    pseudo_labels = generate_pseudo_labels(summaries, taxonomy)
    classifiers, tfidf_vectorizer, label_encoder = train_lightweight_classifiers(summaries, pseudo_labels)
    
    if classifiers and tfidf_vectorizer:
        lr_model, mlp_model, lgb_model = classifiers
        X_test = tfidf_vectorizer.transform(summaries)
        y_test = label_encoder.transform(pseudo_labels)
        
        classification_reports = {
            "Logistic_Regression": classification_report(y_test, lr_model.predict(X_test), 
                                                     labels=np.unique(y_test), 
                                                     target_names=[label_encoder.classes_[i] for i in np.unique(y_test)]),
            "MLP": classification_report(y_test, mlp_model.predict(X_test), 
                                         labels=np.unique(y_test), 
                                         target_names=[label_encoder.classes_[i] for i in np.unique(y_test)]),
            "LightGBM": classification_report(y_test, np.argmax(lgb_model.predict(X_test.toarray()), axis=1), 
                                              labels=np.unique(y_test), 
                                              target_names=[label_encoder.classes_[i] for i in np.unique(y_test)])
        }
        
        save_results(taxonomy, evaluation_metrics, classification_reports)
        
        return classifiers, tfidf_vectorizer, taxonomy, label_encoder
    else:
        print("Error in training the classifiers!")
        return None, None, None, None

if __name__ == "__main__":
    folder = "./climate_plans" # This is where I saved the climate plans in the directory locally
    use_case = "Identify tech-enabled solutions for climate adaptation and resilience"
    main(folder, use_case)
