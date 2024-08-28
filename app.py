import streamlit as st
import pandas as pd
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

def get_first_image_url(urls):
    return urls.split('|')[0]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Replace these paths with the paths to your images in Picture folder
banner_image_path = r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\banner.png"
logo_image_path = r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\logo.png"

encoded_banner = encode_image(banner_image_path)
encoded_logo = encode_image(logo_image_path)

# Load the dataset and model
df = pd.read_csv(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\processed_Women_clothing_with_sentiment.csv") #replace with the processed women clothing data file path
df_men = pd.read_csv(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\processed_Men_with_sentiment_u.csv")  #replace with the processed men clothing data file path
model_men = joblib.load(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\sentiment_model_m_u.pkl")  #replace with the men sentiment analysis model file path
vectorizer_men = joblib.load(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\vectorizer_m_u.pkl")  #replace with the men vectorizer model file path
model = joblib.load(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\sentiment_model.pkl")  #replace with the women sentiment analysis model file path
vectorizer = joblib.load(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\vectorizer.pkl")  #replace with the women vectorizer model file path

def Home_page():
    # Add custom CSS for the header
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .banner {{
            background-image: url("data:image/png;base64,{encoded_banner}");
            background-size: cover;
            background-position: center;
            height: calc(100vh - 150px);
            text-align: center;
            color: white;
            font-size: 36px;
            margin-top: 150px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display the content section
    st.markdown("""
        <div class="banner">
        </div>
        <div class="content">
            <h2>Discover Fashion Insights</h2>
            <p>TrendFusion Analytics is your go-to platform for insightful analysis of clothing and apparel trends. Our application leverages advanced sentiment analysis and data analytics to provide you with valuable insights into the latest fashion trends. Whether you're a fashion retailer, designer, or enthusiast, TrendFusion Analytics helps you stay ahead of the curve.</p>
            <p>Explore our features:</p>
            <ul>
                <li><b>Trend Analysis:</b> Discover top-rated products, most reviewed items, trending colors, and brands.</li>
                <li><b>Product Performance Dashboard:</b> Gain insights into the performance of various products and categories.</li>
            </ul>
            <p>Get started by navigating through the links in the header. Enjoy your journey through the world of fashion analytics!</p>
        </div>
    """, unsafe_allow_html=True)

def create_clickable_url(product_name, url):
    return f'<a href="{url}" target="_blank" style="text-decoration: none; color: #0A122A;">{product_name}</a>'

def display_product_list(products):
    for index, row in products.iterrows():
        image_url = get_first_image_url(row['medium'])
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 10px; text-align: center;">
                <img src="{image_url}" width="100" style="border-radius: 8px; object-fit: cover; margin-bottom: 10px;"/>
                <h4 style="margin: 5px 0;">{create_clickable_url(row['product_name'], row['product_url'])}</h4>
                <p style="margin: 5px 0;">Click on the product name to view</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

def get_first_image_url(image_urls):
    return image_urls.split('|')[0]

def normalize_sentiment(scores):
    return (scores - scores.min()) / (scores.max() - scores.min()) * 100

def display_trends_women(df, sort_by):
    df = df.rename(columns={'predicted_sentiment': 'customer_sentiment_score'})
    df['normalized_sentiment'] = normalize_sentiment(df['customer_sentiment_score'])
    
    color_mapping = {
        'Red': '#FF0000',
        'Black': '#000000',
        'Pink': '#FFC0CB',
        'Green': '#008000',
        'Blue': '#0000FF',
        'Grey': '#808080',
        'Navy Blue': '#000080',
        'White': '#FFFFFF',
        'Orange': '#FFA500',
        'Yellow': '#FFFF00'
    }

    if sort_by == 'Top Choice':
        top_choice = df.groupby('product_name').agg({
            'customer_sentiment_score': 'mean',
            'medium': 'first',
            'product_url': 'first'
        }).reset_index().sort_values(by='customer_sentiment_score', ascending=False).head(10)
        
        display_product_list(top_choice)

    elif sort_by == 'Most Reviewed':
        review_counts = df.groupby('product_name').size().reset_index(name='review_count')
        df_with_counts = pd.merge(df, review_counts, on='product_name', how='left')

        most_reviewed = df_with_counts.groupby('product_name').agg({
            'review_count': 'first',
            'medium': 'first',
            'product_url': 'first'
        }).reset_index().sort_values(by='review_count', ascending=False).head(10)
        
        st.write("Most Reviewed Products:")
        display_product_list(most_reviewed)

    elif sort_by == 'Top Trending Colors':
        top_colors = df.groupby('colour')['normalized_sentiment'].sum().reset_index().sort_values(by='normalized_sentiment', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        colors = [color_mapping.get(color, '#808080') for color in top_colors['colour']]
        plt.bar(top_colors['colour'], top_colors['normalized_sentiment'], color=colors, edgecolor='black')
        plt.title('Top Trending Colors by Sentiment')
        plt.xlabel('Color')
        plt.ylabel('Customer Sentiment Score (Normalized)')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the top 10 colours from the most preferred to the least preferred colours by the customers
        </div>
        """,
        unsafe_allow_html=True
    )

    elif sort_by == 'Top Trending Brands':
        top_brands = df.groupby('brand')['normalized_sentiment'].sum().reset_index().sort_values(by='normalized_sentiment', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        plt.bar(top_brands['brand'], top_brands['normalized_sentiment'], color='#698F3F', edgecolor='black')
        plt.title('Top Trending Brands by Sentiment')
        plt.xlabel('Brand')
        plt.ylabel('User Satisfaction Score (Normalized)')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif sort_by == 'Price vs Sentiment':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sales_price', y='normalized_sentiment', data=df)
        plt.title('Price vs Sentiment')
        plt.xlabel('Sales Price')
        plt.ylabel('User Satisfaction Scale (Normalized)')
        st.pyplot(plt)

def render_trend_analysis_page_women(df, encoded_logo):
    st.title("Trend Analysis - Women's Clothing")
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
        <div class="banner">
            <div>Discover the Latest Trends in Women's Fashion</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    sort_by = st.selectbox('Sort By:', ['Top Choice', 'Most Reviewed', 'Top Trending Colors', 'Top Trending Brands', 'Price vs Sentiment'])
    display_trends_women(df, sort_by)
    st.markdown('</div>', unsafe_allow_html=True)


def render_trend_analysis_page_men(df, encoded_logo):
    st.title("Trend Analysis - Men's Clothing")
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
        <div class="banner">
            <div>Discover the Latest Trends in Men's Fashion</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)
    sort_by = st.selectbox('Sort By:', ['Top Choice', 'Most Reviewed', 'Top Trending Colors', 'Top Trending Brands', 'Price vs Sentiment'])
    display_trends_men(df_men, sort_by)
    st.markdown('</div>', unsafe_allow_html=True)

def display_trends_men(df_men, sort_by):
    df_men = df_men.rename(columns={'predicted_sentiment': 'customer_sentiment_score'})
    df_men['normalized_sentiment'] = normalize_sentiment(df_men['customer_sentiment_score'])

    color_mapping = {
        'Red': '#FF0000',
        'Black': '#000000',
        'Pink': '#FFC0CB',
        'Green': '#008000',
        'Blue': '#0000FF',
        'Grey': '#808080',
        'Navy Blue': '#000080',
        'White': '#FFFFFF',
        'Orange': '#FFA500',
        'Yellow': '#FFFF00'
    }
    if sort_by == 'Top Choice':
        top_choice = df_men.groupby('product_name').agg({
            'customer_sentiment_score': 'mean',
            'medium': 'first',
            'product_url': 'first'
        }).reset_index().sort_values(by='customer_sentiment_score', ascending=False).head(10)
        
        display_product_list(top_choice)

    elif sort_by == 'Most Reviewed':
        review_counts = df_men.groupby('product_name').size().reset_index(name='review_count')
        df_with_counts = pd.merge(df_men, review_counts, on='product_name', how='left')

        most_reviewed = df_with_counts.groupby('product_name').agg({
            'review_count': 'first',
            'medium': 'first',
            'product_url': 'first'
        }).reset_index().sort_values(by='review_count', ascending=False).head(10)
        
        st.write("Most Reviewed Products:")
        display_product_list(most_reviewed)

    elif sort_by == 'Top Trending Colors':
        top_colors = df_men.groupby('colour')['normalized_sentiment'].sum().reset_index().sort_values(by='normalized_sentiment', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        colors = [color_mapping.get(color, '#808080') for color in top_colors['colour']]
        plt.bar(top_colors['colour'], top_colors['normalized_sentiment'], color=colors, edgecolor='black')
        plt.title('Top Trending Colors by Sentiment')
        plt.xlabel('Color')
        plt.ylabel('Customer Sentiment Score (Normalized)')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the top 10 colours from the most preferred to the least preferred colours by the customers
        </div>
        """,
        unsafe_allow_html=True
    )

    elif sort_by == 'Top Trending Brands':
        top_brands = df_men.groupby('brand')['normalized_sentiment'].sum().reset_index().sort_values(by='normalized_sentiment', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        plt.bar(top_brands['brand'], top_brands['normalized_sentiment'], color='#698F3F', edgecolor='black')
        plt.title('Top Trending Brands by Sentiment')
        plt.xlabel('Brand')
        plt.ylabel('User Satisfaction Score (Normalized)')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif sort_by == 'Price vs Sentiment':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sales_price', y='normalized_sentiment', data=df_men)
        plt.title('Price vs Sentiment')
        plt.xlabel('Sales Price')
        plt.ylabel('User Satisfaction Scale (Normalized)')
        st.pyplot(plt)


def render_product_performance_page_women(df, encoded_logo):
    """Renders the product performance page specifically for women's products."""
    st.title("Product Performance Dashboard - Women")
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
        <div class="banner">
            <div>Explore the Product Performance for Women</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    product_performance_dashboard(df)

    st.markdown('</div>', unsafe_allow_html=True)

def render_product_performance_page_men(df, encoded_logo):
    """Renders the product performance page specifically for men's products."""
    st.title("Product Performance Dashboard - Men")
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
        <div class="banner">
            <div>Explore the Product Performance for Men</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    # Render dashboard for men
    product_performance_dashboard(df_men)

    st.markdown('</div>', unsafe_allow_html=True)

def product_performance_dashboard(filtered_df):
    # Ensure required columns exist in the DataFrame
    required_columns = {'product_name', 'predicted_sentiment', 'sales_price'}
    if not required_columns.issubset(filtered_df.columns):
        st.error("The dataset is missing required columns: 'product_name', 'predicted_sentiment', or 'sales_price'.")
        return

    # Get the top 10 products by sentiment
    top_10_products = (
        filtered_df
        .groupby('product_name')
        .agg({
            'predicted_sentiment': 'mean',
            'sales_price': 'mean',
        })
        .reset_index()
        .sort_values(by='predicted_sentiment', ascending=False)
    )

    # Scatter Plot: Sales Price vs Sentiment
    st.subheader("Sales Price vs Customer Sentiment")
    plt.figure(figsize=(10, 6))
    plt.scatter(top_10_products['sales_price'], top_10_products['predicted_sentiment'], s=32, alpha=0.8)
    plt.title('Sales Price vs Customer Sentiment')
    plt.xlabel('Sales Price')
    plt.ylabel('Customer Sentiment')
    plt.gca().spines[['top', 'right']].set_visible(False)  # Hide top and right spines
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the price range preferred by the customers
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

    # Add a line after the scatter plot
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    # Call visualization functions with lines after each
    avg_sentiment_by_brand(filtered_df)
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    avg_price_by_brand(filtered_df)
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    top_10_products_by_sentiment(filtered_df)
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    sentiment_by_category(filtered_df)
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)



def avg_sentiment_by_brand(filtered_df, top_n=10):
    st.subheader(f"Top {top_n} and Bottom {top_n} Average Sentiment by Brand")
    
    # Calculate the average sentiment by brand
    avg_sentiment = filtered_df.groupby('brand')['predicted_sentiment'].mean().reset_index().sort_values(by='predicted_sentiment', ascending=False)
    
    # Get the top N and bottom N brands by average sentiment
    top_brands = avg_sentiment.head(top_n)
    bottom_brands = avg_sentiment.tail(top_n)
    
    # Combine top and bottom brands for plotting
    combined_brands = pd.concat([top_brands, bottom_brands])
    combined_brands['Position'] = ['Top'] * len(top_brands) + ['Bottom'] * len(bottom_brands)
    
    # Plotting both top and bottom brands
    plt.figure(figsize=(14, 8))
    sns.barplot(x='predicted_sentiment', y='brand', hue='Position', data=combined_brands, palette={'Top': '#698F3F', 'Bottom': '#0A122A'})
    plt.title(f'Top {top_n} and Bottom {top_n} Brands by Average Sentiment')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Brand')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the top ten and bottom ten brands most preferred by the customers.
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

def avg_price_by_brand(filtered_df, top_n=10):
    st.subheader(f"Top {top_n} and Bottom {top_n} Average Sales Price by Brand")
    
    # Calculate the average sales price by brand
    avg_price = filtered_df.groupby('brand')['sales_price'].mean().reset_index().sort_values(by='sales_price', ascending=False)
    
    # Get the top N and bottom N brands by average sales price
    top_brands = avg_price.head(top_n)
    bottom_brands = avg_price.tail(top_n)
    
    # Combine top and bottom brands for plotting
    combined_brands = pd.concat([top_brands, bottom_brands])
    combined_brands['Position'] = ['Top'] * len(top_brands) + ['Bottom'] * len(bottom_brands)
    
    # Plotting both top and bottom brands
    plt.figure(figsize=(14, 8))
    sns.barplot(x='sales_price', y='brand', hue='Position', data=combined_brands, palette={'Top': '#698F3F', 'Bottom': '#0A122A'})
    plt.title(f'Top {top_n} and Bottom {top_n} Brands by Average Sales Price')
    plt.xlabel('Average Sales Price')
    plt.ylabel('Brand')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the average sales price of each top ten and bottom 10 brands
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

def top_10_products_by_sentiment(filtered_df):
    st.subheader("Top 10 and Bottom 10 Products by Sentiment")
    
    # Calculate the average sentiment by product
    avg_sentiment = filtered_df.groupby('product_name')['predicted_sentiment'].mean().reset_index().sort_values(by='predicted_sentiment', ascending=False)
    
    # Get the top 10 and bottom 10 products by average sentiment
    top_products = avg_sentiment.head(10)
    bottom_products = avg_sentiment.tail(10)
    
    # Plotting top products
    plt.figure(figsize=(12, 6))
    sns.barplot(x='predicted_sentiment', y='product_name', data=top_products, color='#698F3F')
    plt.title('Top 10 Products by Sentiment')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Product Name')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the Top 10 Products by Sentiment
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

    # Plotting bottom products
    plt.figure(figsize=(12, 6))
    sns.barplot(x='predicted_sentiment', y='product_name', data=bottom_products, color='#0A122A')
    plt.title('Bottom 10 Products by Sentiment')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Product Name')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the Bottom 10 Products by Sentiment
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

def sentiment_by_category(filtered_df, top_n=10):
    st.subheader(f"Sentiment by Category (Top {top_n})")
    avg_sentiment = filtered_df.groupby('child_category')['predicted_sentiment'].mean().reset_index().sort_values(by='predicted_sentiment', ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='predicted_sentiment', y='child_category', data=avg_sentiment, palette='viridis')
    plt.title('Average Sentiment by Category')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Category')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown(
        """
        <div style='font-size: 18px; font-weight: bold; margin-top: 20px;'>
            Insights: This depicts the top 10 Categories of Products by Sentiment
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close()

def render_about_us_page():
    st.title("About Us")
    st.markdown(f"""
        <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
        }}
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #804649;
            padding: 10px;
            z-index: 1000;
            box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
            height: 150px;
        }}
        .logo {{
            height: 125px;
            border-radius: 50%;
        }}
        .nav-links a {{
            color: #FBFAF8;
            text-decoration: none;
            padding: 2px 20px;
            border-radius: 4px;
            font-size: 18px;
        }}
        .nav-links a:hover {{
            background-color: #0A122A;
        }}
        .large-logo {{
            margin-top: 20px;
        }}
        .content {{
            padding: 20px;
            font-family: 'Times New Roman', Times, serif;
            color: #0A122A;
            font-size: 20px;
            line-height: 1.6;
        }}
        </style>
        <div class="header">
            <img src="data:image/png;base64,{encoded_logo}" class="logo" alt="Logo">
            <div class="nav-links">
                <a href="?page=home">Home</a>
                <a href="?page=trend_analysis">Trend Analysis</a>
                <a href="?page=product_performance">Product Performance</a>
                <a href="?page=about_us">About Us</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("""
**Project Overview:**

TrendFusion Analytics is an innovative web application designed to provide comprehensive insights into clothing and apparel trends. Utilizing advanced data science techniques, our application delivers valuable analytics on sentiment, product performance, and market trends, helping merchants make informed decisions based on real-time data.

**Methodology:**

Our approach involves leveraging historical data scraped from eCommerce platforms to analyze and visualize key trends. The core methodologies employed in TrendFusion Analytics include:

* **Sentiment Analysis:** We use natural language processing (NLP) to assess customer sentiments from reviews. This helps in understanding consumer preferences and opinions.
* **Trend Analysis:** By examining various parameters such as brand performance and product categories, we identify emerging trends in the market.
* **Data Visualization:** Our application presents data through interactive graphs and dashboards, making it easier to interpret and act on the insights.
        
**Benefits:**

* **Enhanced Decision-Making:** Merchants can make data-driven decisions based on comprehensive trend and sentiment analysis.
* **Market Insights:** Gain insights into top brands, trending colors, and product performance, helping to align marketing strategies and inventory management.
* **User-Friendly Interface:** The application features an intuitive interface, making it accessible for users with varying levels of technical expertise.

**Developers:**

This project is developed by a dedicated team of students from **K S School of Engineerning and Management**.
             
Under the guidance of **Assosiate Prof. Manjunath T K, Project Coordinator** and **Assistant Prof. Tanushree Mohapatra, Mentor**
             
The team members include:

**Vaishnavi Dayanand**
             
**Naina R Koushik**
             
**Priya G S**
              
Together, we are committed to delivering a robust tool that provides valuable insights and supports data-driven decision-making in the apparel industry.
    """)
# replace with the logo image path in pictures folder
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{encoded_large_logo}" class="large-logo" alt="TrendFusion Analytics Logo" width="500">
        </div>
        """.format(encoded_large_logo=encode_image(r"C:\Users\TrendFusionAnalytics\Desktop\TrendFusion Analytics\logo.png")),
        unsafe_allow_html=True,
    )
    
def main():
    st.set_page_config(page_title="TrendFusion Analytics", layout="wide")
    
    # Get the selected page from the URL
    page = st.experimental_get_query_params().get('page', ['home'])[0]

    if page == 'home':
        Home_page()
    elif page == 'trend_analysis':
        # Create dropdown for gender selection
        st.markdown("""
            <style>
            .trend-analysis-dropdown {{
                margin: 20px;
                font-family: 'Times New Roman', Times, serif;
            }}
            </style>
        """, unsafe_allow_html=True)
        gender = st.selectbox("Select Clothing Type", ['Women', 'Men'], key='gender_selection')
        if gender == 'Women':
            render_trend_analysis_page_women(df,encoded_logo)
        elif gender == 'Men':
            render_trend_analysis_page_men(df_men,encoded_logo)
    elif page == 'product_performance':
        # Create dropdown for gender selection
        st.markdown("""
            <style>
            .trend-analysis-dropdown {{
                margin: 20px;
                font-family: 'Times New Roman', Times, serif;
            }}
            </style>
        """, unsafe_allow_html=True)
        gender = st.selectbox("Select Clothing Type", ['Women', 'Men'], key='gender_selection')
        if gender == 'Women':
            render_product_performance_page_women(df, encoded_logo)
        elif gender == 'Men':
            render_product_performance_page_men(df, encoded_logo)
        
    elif page == 'about_us':
        render_about_us_page()

if __name__ == '__main__':
    main()