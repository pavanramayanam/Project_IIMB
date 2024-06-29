import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import ast
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import google.generativeai as genai

# Load the fashion dataset and embeddings
fashion_data = pd.read_excel("data/embedded_sample_data_v2.xlsx")  # 1119 records a.o.n

api_key = "AIzaSyCNK89hFIgmo33N0SGNzy7jZV1aLraQYBo"
# Configure API key (replace with your actual key)
genai.configure(api_key=api_key)

# Function to embed text using Google GenAI
@lru_cache(maxsize=None)
def embed_text(text):
    return np.array(genai.embed_content(model='models/text-embedding-004', content=text)['embedding']).reshape(1, -1)

# Function to find top similar products based on embeddings
def find_top_similar_products(df, search_embedding, top_n):
    embeddings = np.array(df['embeddings'].apply(ast.literal_eval).tolist())
    similarities = cosine_similarity(search_embedding, embeddings)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return df.iloc[top_indices]

# Function to display product information and image
def display_product_info_and_image(product, col1, col2):
    image = Image.open(product['first_image_path'])
    col1.image(image, caption="", width=200)
    col2.write(f"**Name:** {product['name']}")
    description = product['product_details_description'].replace('_x000D_', '\n')
    col2.write(f"**Description:** {description}", unsafe_allow_html=True)

# Function to display similar products
def display_similar_products(similar_products, columns):
    st.write(similar_products.drop(columns=['id','product_details_description','image_paths','first_image_path','to_embed','embeddings']))
    for idx, (_, row) in enumerate(similar_products.iterrows()):
        col = columns[idx % len(columns)]
        image = Image.open(row['first_image_path'])
        col.image(image, caption="", width=100)
        col.write(f"**Name:** {row['name']}")

def main():
    st.title("Fashion Recommender System")
    top_n = 5 # top n recommendation
    col1, col2 = st.columns(2)

    # Display count of men and women apparels
    col1.write('**Men/Women apparels count:**')
    col1.write(fashion_data.gender.value_counts())

    # Display types of apparels
    col2.write('**Type of Apparels we have:**')
    col2.write(fashion_data['articleType'].value_counts().reset_index())

    # Input for searching products
    query = st.text_input("Search for products:")
    if st.button("Search"):
        # Embed the search query
        search_embedding = embed_text(query)
        
        # Find top similar products based on the search embedding
        results = find_top_similar_products(fashion_data, search_embedding, top_n)
        
        # Display top similar products
        st.header("Products:")
        st.write(results.drop(columns=['id','product_details_description','image_paths','first_image_path','to_embed','embeddings']))

        # Display product information and similar products for each top similar product
        for _, row in results.iterrows():
            col1, col2 = st.columns(2)
            display_product_info_and_image(row, col1, col2)
            with st.expander(f":rainbow[Similar products for {row['name']}]"):
                # Filter data based on product attributes and calculate similar products
                filtered_data = fashion_data[(fashion_data['id'] != row['id']) & 
                                            (fashion_data['gender'] == row['gender']) & 
                                            (fashion_data['articleType'] == row['articleType'])]
                if filtered_data.empty:
                    similar_products = find_top_similar_products(fashion_data, 
                                                                np.array(ast.literal_eval(row['embeddings'])).reshape(1, -1), 
                                                                top_n)
                else:
                    similar_products = find_top_similar_products(filtered_data, 
                                                                np.array(ast.literal_eval(row['embeddings'])).reshape(1, -1), 
                                                                top_n)
                    if len(filtered_data) < top_n:
                        remaining_count = top_n - len(filtered_data)
                        remaining_data = fashion_data[~fashion_data['id'].isin(filtered_data['id'])]
                        similar_products_from_remaining = find_top_similar_products(remaining_data, 
                                                                                    np.array(ast.literal_eval(row['embeddings'])).reshape(1, -1), 
                                                                                    remaining_count)
                        similar_products = pd.concat([filtered_data, similar_products_from_remaining])

                # Display similar products
                display_similar_products(similar_products, st.columns(top_n))

if __name__ == "__main__":
    main()