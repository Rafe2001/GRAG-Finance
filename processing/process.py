import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from transformers import pipeline 
import re
from collections import defaultdict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NLPPipeline:
    def __init__(self):
        """Initialize NLP models and pipelines"""
        # Load SpaCy model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            logging.info("Downloading SpaCy model...")
            spacy.cli.download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg")

        # Initialize transformers pipeline for sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis",
                                           model = "ProsusAI/finbert")

        # Custom financial metrics patterns
        self.metric_patterns = {
            'revenue': r'\$?\d+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T)',
            'growth': r'\d+(?:\.\d+)?%\s*(?:increase|decrease|growth)',
            'market_share': r'\d+(?:\.\d+)?%\s*(?:market share|share)',
        }

        # Define relationship patterns
        self.relationship_patterns = [
            (r'(.*?)\s*acquired\s*(.*)', 'ACQUIRED'),
            (r'(.*?)\s*partnered with\s*(.*)', 'PARTNERED'),
            (r'(.*?)\s*invested in\s*(.*)', 'INVESTED'),
            (r'(.*?)\s*announced\s*(.*)', 'ANNOUNCED'),
            (r'(.*?)\s*launched\s*(.*)', 'LAUNCHED'),
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            # Clean the entity text
            clean_ent = ent.text.strip()

            # Categorize entities
            if ent.label_ in ['ORG']:
                entities['organizations'].append(clean_ent)
            elif ent.label_ in ['PERSON']:
                entities['people'].append(clean_ent)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(clean_ent)
            elif ent.label_ in ['MONEY', 'PERCENT']:
                entities['metrics'].append(clean_ent)

        # Extract financial metrics using custom patterns
        for metric_type, pattern in self.metric_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            entities[f'financial_{metric_type}'] = [match.group(0) for match in matches]

        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return dict(entities)

    def extract_relationships(self, text: str) -> List[Dict]:
        """
        Extract relationships between entities

        Args:
            text: Input text

        Returns:
            List of relationship dictionaries
        """
        relationships = []
        doc = self.nlp(text)
        sentences = list(doc.sents)

        for sent in sentences:
            sent_text = sent.text.strip()

            # Check for relationship patterns
            for pattern, rel_type in self.relationship_patterns:
                matches = re.finditer(pattern, sent_text, re.IGNORECASE)

                for match in matches:
                    # Extract entities involved in the relationship
                    subject = match.group(1).strip() if match.group(1) else ''
                    object_ = match.group(2).strip() if match.group(2) else ''

                    if subject and object_:
                        relationships.append({
                            'subject': subject,
                            'relationship': rel_type,
                            'object': object_,
                            'sentence': sent_text
                        })

        return relationships

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Perform sentiment analysis on text

        Args:
            text: Input text

        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Split text into smaller chunks if too long
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

            # Analyze sentiment for each chunk
            sentiments = [self.sentiment_analyzer(chunk)[0] for chunk in chunks]

            # Aggregate results
            avg_score = np.mean([s['score'] for s in sentiments])
            dominant_label = max(set([s['label'] for s in sentiments]),
                               key=lambda x: [s['label'] for s in sentiments].count(x))

            return {
                'label': dominant_label,
                'score': float(avg_score),
                'confidence': float(max([s['score'] for s in sentiments]))
            }
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}

class TextProcessor:
    def __init__(self):
        """Initialize text processing pipeline"""
        self.nlp_pipeline = NLPPipeline()

    def process_article(self, article: Dict) -> Dict:
        """
        Process a single news article

        Args:
            article: Dictionary containing article data

        Returns:
            Processed article with extracted information
        """
        processed = article.copy()

        # Combine title and content for processing
        full_text = f"{article['title']}. {article['content']}"

        # Extract entities
        processed['extracted_entities'] = self.nlp_pipeline.extract_entities(full_text)

        # Extract relationships
        processed['extracted_relationships'] = self.nlp_pipeline.extract_relationships(full_text)

        # Analyze sentiment
        processed['sentiment'] = self.nlp_pipeline.analyze_sentiment(full_text)

        return processed

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataset of articles

        Args:
            df: DataFrame containing articles

        Returns:
            Processed DataFrame with extracted information
        """
        processed_articles = []

        for _, row in df.iterrows():
            try:
                processed_article = self.process_article(row.to_dict())
                processed_articles.append(processed_article)
                logging.info(f"Processed article: {row['title'][:50]}...")
            except Exception as e:
                logging.error(f"Error processing article: {str(e)}")
                continue

        return pd.DataFrame(processed_articles)

# main.py
def main():
    # Load the collected news data
    try:
        df = pd.read_csv('tech_news.csv')
        logging.info(f"Loaded {len(df)} articles for processing")
    except FileNotFoundError:
        logging.error("News data file not found. Please run Phase 1 first.")
        return

    # Initialize processor
    processor = TextProcessor()

    # Process the dataset
    processed_df = processor.process_dataset(df)

    # Save processed data
    processed_df.to_json('data/processed_news.json', orient='records', indent=2)

    # Print summary statistics
    print("\nNLP Processing Summary:")
    print("----------------------")
    print(f"Total articles processed: {len(processed_df)}")

    # Sentiment distribution
    sentiment_dist = processed_df['sentiment'].apply(lambda x: x['label']).value_counts()
    print("\nSentiment Distribution:")
    print(sentiment_dist)

    # Entity statistics
    total_entities = sum([len(ents) for ents in processed_df['extracted_entities']])
    print(f"\nTotal entities extracted: {total_entities}")

    # Relationship statistics
    total_relationships = sum([len(rels) for rels in processed_df['extracted_relationships']])
    print(f"Total relationships extracted: {total_relationships}")

if __name__ == "__main__":
    main()
    