# -*- coding: utf-8 -*-
"""Booking_Analytics_&_QA_System.ipynb
import gdown
import os

faiss_url = "https://drive.google.com/uc?id=1tV_-8UvgAbXsqKkfiRan9_rAUm-lvAlE"
faiss_path = "booking_index.faiss"

if not os.path.exists(faiss_path):
    print("Downloading FAISS index...")
    gdown.download(faiss_url, faiss_path, quiet=False)
else:
    print("FAISS index already exists.")





# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# 1. Data Collection & Preprocessing

# Assuming you have a Kaggle dataset named 'hotel_bookings.zip' in your Colab environment
!unzip archive.zip

# Load the dataset using pandas
df = pd.read_csv('/content/llm-booking-analytics/hotel_bookings.csv')  # Replace with your actual file name

# Data Cleaning
# Handle missing values (replace with mean/median or remove rows/columns)
df.fillna(method='ffill', inplace=True)  # Replace missing values with the previous value in the column

# Format inconsistencies
# Example: Convert date columns to datetime objects
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-' + df['arrival_date_day_of_month'].astype(str))

# 2. Analytics & Reporting

# Revenue trends over time
df['revenue'] = df['adr'] * df['stays_in_weekend_nights'] + df['adr'] * df['stays_in_week_nights']
revenue_trends = df.groupby('arrival_date')['revenue'].sum()
plt.plot(revenue_trends.index, revenue_trends.values)
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.title('Revenue Trends Over Time')
plt.show()

# Cancellation rate
cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Geographical distribution of users
user_location_distribution = df.groupby('country')['country'].count()
plt.figure(figsize=(15, 5))
sns.barplot(x=user_location_distribution.index, y=user_location_distribution.values)
plt.ylabel('Number of Bookings')
plt.title('Geographical Distribution of Bookings')
plt.show()

# Booking lead time distribution
plt.hist(df['lead_time'], bins=20)
plt.xlabel('Lead Time (Days)')
plt.ylabel('Frequency')
plt.title('Booking Lead Time Distribution')
plt.show()

# Additional Analytics (example)
# Average booking price per hotel type
average_price_by_hotel = df.groupby('hotel')['adr'].mean()
print(average_price_by_hotel)

import pandas as pd

# Load Booking Data
df = pd.read_csv("hotel_bookings.csv")

# Print column names to verify available data
print("Columns in dataset:", df.columns)

# Identify the correct date column
date_column = None
possible_date_cols = ["date", "reservation_status_date", "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]

for col in possible_date_cols:
    if col in df.columns:
        date_column = col
        break

if date_column is None:
    raise KeyError("No valid date column found in dataset!")

# Convert date column to a unified format if applicable
if date_column == "reservation_status_date":
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

# Ensure 'summary' column exists or create it
df["summary"] = df.apply(lambda row: (
    f"Date: {row[date_column]} "
    f"Hotel: {row['hotel']}, "
    f"Adults: {row['adults']}, "
    f"Children: {row['children']}, "
    f"Babies: {row['babies']}, "
    f"ADR: {row['adr']}"), axis=1)

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer with GPU acceleration
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Convert text data to embeddings efficiently
embeddings = embedding_model.encode(
    df["summary"].tolist(),
    convert_to_numpy=True,
    batch_size=128,
    show_progress_bar=True
)

# Initialize FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS Index
faiss.write_index(index, "booking_index.faiss")

print("FAISS index saved successfully!")

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Correct import for LangChain 0.2.2+

# Load Sentence Transformer with GPU acceleration
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# ✅ Ensure `df["summary"]` exists
if "summary" not in df.columns:
    raise ValueError("The DataFrame does not contain a 'summary' column.")

# ✅ Extract text data
texts = df["summary"].tolist()  # Ensure texts are defined

# Convert text data to embeddings efficiently
embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True,
    batch_size=128,
    show_progress_bar=True
)

# Initialize FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS Index
faiss.write_index(index, "booking_index.faiss")

# ✅ Define `embedding_function` (DO NOT pass `embeddings` explicitly)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Corrected FAISS loading
db = FAISS.from_texts(texts, embedding_function)

# Example Query
query = "Show me bookings canceled in March"
query_embedding = embedding_model.encode([query], convert_to_numpy=True)
results = db.similarity_search_by_vector(query_embedding[0], k=5)

for result in results:
    print(result.page_content)

def answer_query(query):
    # Convert query to lowercase for better matching
    query = query.lower()

    # Identify cancellations on a specific date
    if "canceled on" in query:
        # Extract date from query (assuming format like "canceled on 2017-07-15")
        date_str = query.split("canceled on")[-1].strip()

        try:
            date_obj = pd.to_datetime(date_str)
            filtered_df = df[(df["is_canceled"] == 1) & (df["reservation_status_date"] == date_obj)]

            if filtered_df.empty:
                return f"No bookings were canceled on {date_str}."

            return filtered_df[["hotel", "adults", "children", "babies", "adr"]].to_string(index=False)

        except Exception as e:
            return f"Error processing date: {e}"

    # Find bookings for a specific country
    elif "bookings from" in query:
        country_code = query.split("bookings from")[-1].strip().upper()

        filtered_df = df[df["country"] == country_code]

        if filtered_df.empty:
            return f"No bookings found from {country_code}."

        return filtered_df[["hotel", "adults", "children", "babies", "adr"]].to_string(index=False)

    # Use FAISS for general retrieval-based queries
    query_embedding = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    results = db.similarity_search_by_vector(query_embedding[0], k=5)

    return [result.page_content for result in results]

# Example queries
print(answer_query("Which locations had the highest booking cancellations?"))

import pandas as pd

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in dataset:", df.columns)  # Check column names
    return df

# Function to compute insights
def compute_insights(df):
    insights = {}

    # Check if the correct date column exists
    date_column = None
    possible_date_cols = ["date", "Date", "arrival_date"]  # Adjust based on actual column names
    for col in possible_date_cols:
        if col in df.columns:
            date_column = col
            break

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Example: Total bookings per hotel
    insights["total_bookings_per_hotel"] = df["hotel"].value_counts().to_dict()

    # Example: Find all unique canceled bookings
    if "reservation_status" in df.columns:
        insights["canceled_bookings"] = df[df["reservation_status"] == "Canceled"].to_dict(orient="records")

    return insights

# Function to store insights
def store_insights(insights):
    print("Insights stored successfully:", insights)

if __name__ == "__main__":
    file_path = "/content/hotel_bookings.csv"  # Correct file path
    df = load_data(file_path)
    insights = compute_insights(df)
    store_insights(insights)

import pandas as pd

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce")
    return df

# Function to compute insights
def compute_insights(df):
    insights = {}

    # Total bookings per hotel
    insights["total_bookings_per_hotel"] = df["hotel"].value_counts().to_dict()

    # Find all canceled bookings
    if "reservation_status" in df.columns:
        canceled_df = df[df["reservation_status"] == "Canceled"]
        insights["canceled_bookings"] = canceled_df.to_dict(orient="records")

    # Find bookings canceled on a specific date
    specific_date = "2017-07-01"  # Change to the required date
    canceled_on_date = df[(df["reservation_status"] == "Canceled") &
                          (df["reservation_status_date"] == specific_date)]
    insights["canceled_on_specific_date"] = canceled_on_date.to_dict(orient="records")

    return insights

# Function to store insights
def store_insights(insights):
    print("Insights saved successfully...")
    print(insights)  # Display results

if __name__ == "__main__":
    file_path = "/content/hotel_bookings.csv"  # Correct file path
    df = load_data(file_path)
    insights = compute_insights(df)
    store_insights(insights)

import pandas as pd
import json

# Load dataset
df = pd.read_csv("hotel_bookings.csv")

# Data Cleaning
df.dropna(inplace=True)  # Drop missing values
df["arrival_date"] = pd.to_datetime(df["arrival_date_year"].astype(str) + "-" + df["arrival_date_month"] + "-" + df["arrival_date_day_of_month"].astype(str))

# Convert Timestamp objects to string
df["arrival_date"] = df["arrival_date"].astype(str)

# Precompute Insights
insights = {
    "total_bookings": len(df),
    "total_revenue": df["adr"].sum(),
    "canceled_bookings": df[df["is_canceled"] == 1].to_dict(orient="records"),
    "top_cancellation_locations": df[df["is_canceled"] == 1]["country"].value_counts().to_dict()
}

# Save insights
with open("insights.json", "w") as file:
    json.dump(insights, file, indent=4)

print("Precomputed insights saved!")

import os
print(os.listdir())  # Check available files in the current directory

import shutil
import os
from google.colab import files

# Define folder name
folder_name = "llm-booking-analytics"

# Create a folder
os.makedirs(folder_name, exist_ok=True)

# Move all files to the new folder (excluding system folders)
for file in os.listdir():
    if file not in [folder_name, "sample_data"]:  # Exclude system files/folders
        shutil.move(file, os.path.join(folder_name, file))

# Zip the folder
shutil.make_archive(folder_name, 'zip', folder_name)

# Download the zip file
files.download(folder_name + ".zip")

