# data_collection.py
import os
import requests
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NewsDataCollector:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize API key
        self.news_api_key = "d990044ce4834d8f84dba33bdfed35d3"

        # Base URL
        self.news_api_base_url = "https://newsapi.org/v2"

        # Verify API key is present
        if not self.news_api_key:
            raise ValueError("Missing NEWS_API_KEY. Please check your .env file.")

    def fetch_tech_news(self, companies: List[str], days: int = 30) -> pd.DataFrame:
        """
        Fetch recent tech news for specified companies

        Args:
            companies: List of company names to search for
            days: Number of days of historical news to fetch

        Returns:
            DataFrame containing news articles
        """
        all_articles = []
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        for company in companies:
            try:
                # Construct query for tech-related news
                query = (
                    f'"{company}" AND '
                    '(technology OR tech OR earnings OR revenue OR '
                    'partnership OR acquisition OR AI OR cloud OR innovation)'
                )

                url = f"{self.news_api_base_url}/everything"
                params = {
                    'q': query,
                    'from': from_date,
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'apiKey': self.news_api_key
                }

                response = requests.get(url, params=params)
                response.raise_for_status()

                articles = response.json().get('articles', [])

                # Add company and extracted information to each article
                for article in articles:
                    article['company'] = company
                    article['extracted_metrics'] = self.extract_basic_metrics(article['content'])

                all_articles.extend(articles)
                logging.info(f"Fetched {len(articles)} articles for {company}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching news for {company}: {str(e)}")
                continue

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            return df
        else:
            return pd.DataFrame()

    def extract_basic_metrics(self, content: str) -> Dict:
        """
        Extract basic financial metrics from article content
        This is a simple extraction - you might want to enhance this with better NLP

        Args:
            content: Article content text

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {
            'has_revenue_mention': False,
            'has_growth_mention': False,
            'has_partnership_mention': False,
            'has_acquisition_mention': False
        }

        # Simple keyword checking
        content = content.lower()
        metrics['has_revenue_mention'] = any(word in content for word in ['revenue', 'earnings', 'profit'])
        metrics['has_growth_mention'] = any(word in content for word in ['growth', 'increase', 'grew'])
        metrics['has_partnership_mention'] = any(word in content for word in ['partnership', 'collaborate', 'alliance'])
        metrics['has_acquisition_mention'] = any(word in content for word in ['acquisition', 'acquired', 'bought'])

        return metrics

class DataProcessor:
    @staticmethod
    def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess news articles data

        Args:
            df: DataFrame containing news articles

        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['title', 'publishedAt'])

        # Remove articles with missing content
        df = df.dropna(subset=['content', 'title', 'url', 'description'])

        # Clean text fields
        df['content'] = df['content'].str.replace('\r', ' ').str.replace('\n', ' ')
        df['title'] = df['title'].str.strip()

        # Convert dates to datetime
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])

        # Create derived features
        df['article_length'] = df['content'].str.len()
        df['title_length'] = df['title'].str.len()
        # Extract metrics columns from the dictionary
        metrics_df = pd.json_normalize(df['extracted_metrics'])
        df = pd.concat([df, metrics_df], axis=1)

        return df

# main.py
def main():
    # Initialize collector and processor
    collector = NewsDataCollector()
    processor = DataProcessor()

    # Define target companies
    tech_companies = [
        'Microsoft',
        'Apple',
        'Google',
        'Amazon',
        'Meta',
        'NVIDIA',
        'Tesla'
    ]

    # Collect news data
    news_df = collector.fetch_tech_news(tech_companies)

    # Clean and process the data
    cleaned_news_df = processor.clean_news_data(news_df)

    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save processed data
    cleaned_news_df.to_csv('data/tech_news.csv', index=False)

    # Print summary statistics
    print("\nData Collection Summary:")
    print("-----------------------")
    print(f"Total articles collected: {len(cleaned_news_df)}")
    print("\nArticles per company:")
    print(cleaned_news_df['company'].value_counts())
    print("\nMetrics mention statistics:")
    metrics_cols = [col for col in cleaned_news_df.columns if col.startswith('has_')]
    print(cleaned_news_df[metrics_cols].sum())

if __name__ == "__main__":
    main()

df = pd.read_csv("data/tech_news.csv")
df.shape

df = df.drop(columns = ['urlToImage'], axis = 1)

df = df[df['title']!='[Removed]']
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['title'], inplace=True)
df.reset_index(inplace=True)
df.to_csv("tech_news.csv", index=False)
