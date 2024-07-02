import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import ast
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import google.generativeai as genai
import base64
from io import BytesIO
from PIL import Image
from constants import API_KEY, EMBED_DATA

# Load the fashion dataset and embeddings
fashion_data = pd.read_json(EMBED_DATA)

# Configure API key (replace with your actual key)
genai.configure(api_key=API_KEY)

# Set the page config for wide mode
st.set_page_config(layout="wide", page_title="Fashion Recommender System")

# Function to embed text using Google Gemini
@lru_cache(maxsize=None)
def embed_text(text):
    response = genai.embed_content(model='models/text-embedding-004', content=text)
    return np.array(response['embedding']).reshape(1, -1)

# Function to process image and combine attributes (avoiding duplicates)
def process_image(image):
    generation_config = {"temperature": 0}
    model = genai.GenerativeModel('gemini-pro-vision',generation_config=generation_config)
    attributes = '''Color: Describing the hue, saturation, and brightness of the fabric. 
                    Pattern: Any design or motif on the fabric, such as stripes, polka dots, floral prints, or geometric shapes.
                    Detailing: Any additional decorative elements on the garment, such as lace, embroidery, sequins, ruffles, etc.
                    Neckline: The shape of the opening of the garment around the neck.
                    Sleeve Length: Describing whether the sleeves are long, short, three-quarter length, or sleeveless.
                    Fit: How the garment conforms to the body, such as loose, tight, baggy, or fitted.
                    Style: The overall aesthetic or fashion of the garment, such as casual, formal, vintage, bohemian, sporty, etc.
                    Length: How long or short the garment is, like short-sleeved, knee-length, ankle-length, etc.
                    Functionality: Any special features or practical aspects of the garment, such as pockets, zippers, buttons, adjustable straps, etc.
                    Occasion: Where and when the garment might be worn, such as work, party, casual outing, formal event, etc.
                    Seasonality: Whether the garment is suited for a particular season, such as lightweight for summer or heavy for winter.
                    '''
    try:
        # Prompt for attribute extraction
        prompt = f'''You are the Cloth Expert and Fashion Expert, renowned for your keen eye for detail and deep understanding
        of clothing attributes. Tasked with being the go-to personal style advisor, your mission is to provide invaluable
        guidance to anyone seeking detailed information about clothing attributes. From color coordination to fabric texture,
        you excel in deciphering the nuances of fashion to help individuals make informed decisions and elevate their style.
        Analyze the image to extract relevant attributes. 
        Here are the attributes that you should look for:{attributes}
        An example output could be like:
        Color: Light blue, Pattern: Subtle striped pattern, Detailing: Fine stitching, durable buttons, Sleeve Length: Long sleeves, Fit: Tailored, Style: Versatile
        GIVE ONLY THE ATTRIBUTES THAT ARE RELEVANT FOR THAT PARTICULAR CLOTH
        '''

        # Generate content with Gemini Vision
        response = model.generate_content(contents=[prompt, image])
        image.close()

        # Extract text
        extracted_text = response.text.strip()

        # Process extracted text
        all_attributes = extracted_text.replace('\n',', ') #set(attribute.strip() for attribute in extracted_text.split(","))

        return all_attributes

    except Exception as e:
        print(f"Error processing the provided image: {e}")
        return set()

# Function to find top similar products based on embeddings
def find_top_similar_products(df, search_embedding, top_n):
    embeddings = np.vstack(df['embeddings'].apply(ast.literal_eval).values)
    similarities = cosine_similarity(search_embedding, embeddings)
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    return df.iloc[top_indices]

# Function to display product information and image
def display_product_info_and_image(product):
    col1, col2 = st.columns([1, 2])
    image_data = base64.b64decode(product['image_base64'])
    #bites = BytesIO(base64.b64decode(image_data))
    #aux_im = Image.open(BytesIO(image_data))
    
    # Display the image in Streamlit
    #st.image(image_data, caption='Base64 Image', use_column_width=True)
    
    col1.image(image_data, caption="", width=200)
    col2.write(f"**Name:** {product['name']}")
    col2.write(f"**Gender:** {product['gender']}")
    col2.write(f"**Description:** {product['image_desc']}")        
    # Create a button with JavaScript for opening a new tab
    button = f'''
        <a href="{product['myntra_product_url']}" target="_blank">
            <button style="
                background-color: magenta; 
                border: none; 
                color: white; 
                padding: 10px 25px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 16px;
                cursor: pointer;">
                BUY FROM MYNTRA
            </button>
        </a>
    '''

    col2.markdown(button, unsafe_allow_html=True)
    
# Function to display similar products
def display_similar_products(similar_products):
    columns = st.columns(4)
    for idx, (_, row) in enumerate(similar_products.iterrows()):
        col = columns[idx % len(columns)]
        image_data = base64.b64decode(row['image_base64'])
        col.image(image_data, caption=row['name'], width=150)
        col.write(f"**Gender:** {row['gender']}")
        col.write("**Description:**")
        col.write(f"{row['image_desc']}")
        # Create a button with JavaScript for opening a new tab
        button = f'''
            <a href="{row['myntra_product_url']}" target="_blank">
                <button style="
                    background-color: #4CAF50; 
                    border: none; 
                    color: white; 
                    padding: 10px 25px;
                    text-align: center; 
                    text-decoration: none; 
                    display: inline-block; 
                    font-size: 16px;
                    cursor: pointer;">
                    BUY FROM MYNTRA
                </button>
            </a>
        '''

        col.markdown(button, unsafe_allow_html=True)

# Function to extract color from text
def extract_color_from_text(text):
    colors = ["blue", "red", "green", "yellow", "black", "white", "pink", "purple", "orange", "grey", "brown"]
    text_lower = text.lower()
    for color in colors:
        if color in text_lower:
            return color
    return None

# Function to filter products based on color mentioned in the query
def filter_products_by_color(df, query):
    query_color = extract_color_from_text(query)
    if query_color:
        df['inferred_color'] = df['image_desc'].apply(extract_color_from_text)
        return df[df['inferred_color'] == query_color]
    return df

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>STYLE ON THE GO...</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: blue;'>By M.I.D.A.S</h5>", unsafe_allow_html=True)

    with st.expander(":red[_Disclaimer_]"):
        st.write('''
                We source our data from myntra.com. Our aim with these fashion recommendations is to guide you towards products based on the following criteria:
                1. Product Name: The specific name of the item.
                2. Image Description, including:
                    - Color: Describing the tone, saturation, and brightness of the fabric.
                    - Pattern: Any designs or motifs present on the fabric, such as stripes, polka dots, floral prints, or geometric shapes.
                    - Fit: How the garment conforms to the body, whether it's loose, tight, baggy, or fitted.
                    - Style: The overall aesthetic or fashion sense embodied by the garment, encompassing categories like casual, formal, vintage, bohemian, sporty, etc.
                    - Length: The garment's measurement in terms of how long or short it is, including descriptors like short-sleeved, knee-length, ankle-length, etc.
                    - Neckline: The shape of the garment's opening around the neck.
                    - Sleeve Length: Description of the sleeves, indicating if they are long, short, three-quarter length, or sleeveless.
                    - Detailing: Any additional decorative elements adorning the garment, such as lace, embroidery, sequins, ruffles, etc.
                    - Functionality: Special features or practical aspects of the garment, like pockets, zippers, buttons, adjustable straps, etc.
                    - Occasion: Suitable events or scenarios for wearing the garment, such as work, parties, casual outings, formal events, etc.
                    - Seasonality: Whether the garment is intended for a particular season, such as lightweight fabrics for summer or heavier materials for winter."
                ''')
        
        st.write("We have the data as per the following counts:")
        col1, col2, col3 = st.columns(3)
        col1.write("1. Gender:")
        col1.write(fashion_data['gender'].value_counts())
        col2.write("2. Apparel Type:")
        col2.write(fashion_data['articleType'].value_counts())
        col3.write("3. Brand:")
        col3.write(fashion_data['brand'].value_counts())
        
    top_n = 6  # top n recommendation
    st.header(f"Top {top_n} recommendations:")
    
    with st.sidebar:
        st.header("Filters")
        gender_filter = st.selectbox("Gender", ["All", "Men", "Women"])
        apparel_filter = st.selectbox("Apparel Type", ["All"] + list(fashion_data['articleType'].unique()))
        brand_filter = st.selectbox("Brand", ["All"] + list(fashion_data['brand'].unique()))

    filtered_data = fashion_data.copy()
    if gender_filter != "All":
        filtered_data = filtered_data[filtered_data['gender'] == gender_filter]
    if apparel_filter != "All":
        filtered_data = filtered_data[filtered_data['articleType'] == apparel_filter]
    if brand_filter != "All":
        filtered_data = filtered_data[filtered_data['brand'] == brand_filter]

    query = st.text_input("Search for products:")
    uploaded_image = st.file_uploader("Or upload an image:", type=["jpg", "jpeg", "png"])
    
    button = st.button("Relevant Search")

    if button:
        if uploaded_image:
            image = Image.open(uploaded_image)
            query = process_image(image)
            st.write(f'Image description: {query}')
        else:
            # Filter products based on color mentioned in the query
            filtered_data = filter_products_by_color(filtered_data, query)
            
        # Embed the search query using Google Gemini
        search_embedding = embed_text(query)
        
        # Find top similar products based on the search embedding
        results = find_top_similar_products(filtered_data, search_embedding, top_n)

        # Display top similar products
        st.header("Products:")

        # Display product information and similar products for each top similar product
        for _, row in results.iterrows():
            display_product_info_and_image(row)
            with st.expander(f":red[Similar products for {row['name']}]"):
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
                display_similar_products(similar_products)

if __name__ == "__main__":
    main()
