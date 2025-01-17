from neo4j import GraphDatabase
import pandas as pd
import json
import logging
from typing import List, Dict, Optional
import numpy as np
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import GraphQAChain
import os
from groq import Groq


logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

neo4j_uri = "bolt://44.202.210.236:7687"
neo4j_user = "neo4j"
neo4j_password = "sisters-car-recipients"
groq_api_key = "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"


class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.setup_constraints()

    def setup_constraints(self):
        """Set up Neo4j constraints"""
        with self.driver.session() as session:
            # Create constraints for unique nodes
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE"
            ]

            for constraint in constraints:
                session.run(constraint)

    def create_company_node(self, tx, company: str):
        """Create a company node"""
        query = """
        MERGE (c:Company {name: $company})
        RETURN c
        """
        tx.run(query, company=company)

    def create_article_node(self, tx, article: Dict):
        if article.get("url") and pd.notna(article['url']):
            # Create article node
            query = """
            MERGE (a:Article {url: $url})
            SET a.title = $title,
                a.publishedAt = $publishedAt,
                a.sentiment = $sentiment
            WITH a
            MATCH (c:Company {name: $company})
            MERGE (c)-[:MENTIONED_IN]->(a)
            """
            tx.run(query,
                url=article['url'],
                title=article['title'],
                publishedAt=article['publishedAt'],
                sentiment=article['sentiment']['label'],
                company=article['company']
            )

            # Handle co-mentioned companies
            companies_in_article = article.get('extracted_entities', {}).get('organizations', [])
            if isinstance(companies_in_article, list):
                for i, company1 in enumerate(companies_in_article):
                    for company2 in companies_in_article[i+1:]:
                        # Create relationship query directly
                        relationship_query = """
                        MATCH (c1:Company {name: $subject})
                        MATCH (c2:Company {name: $object})
                        MERGE (c1)-[r:RELATIONSHIP {type: $relationship, sentence: $sentence}]->(c2)
                        """
                        tx.run(relationship_query,
                            subject=company1,
                            object=company2,
                            relationship="CO_MENTIONED",
                            sentence=article['content']
                        )
            else:
                logging.warning(f"Skipping article with invalid organizations format: {article}")

            # Handle financial metrics
            if "financial_revenue" in article['extracted_entities'] and article['extracted_entities']['financial_revenue']:
                for revenue in article['extracted_entities']['financial_revenue']:
                    metric_query = """
                    MATCH (c:Company {name: $company})
                    MERGE (m:Metric {value: $value, type: $type, date: $date})
                    MERGE (c)-[:HAS_METRIC]->(m)
                    """
                    tx.run(metric_query,
                        company=article['company'],
                        value=revenue,
                        type='revenue',
                        date=article['publishedAt']
                    )
            logging.info(f"Processed article: {article['title']}")
        else:
            logging.warning(f"Skipping article with missing URL: {article}")

    def create_relationship_nodes(self, tx, relationship: Dict):
        """Create relationship between entities"""
        query = """
        MATCH (c1:Company {name: $subject})
        MATCH (c2:Company {name: $object})
        MERGE (c1)-[r:RELATIONSHIP {type: $relationship, sentence: $sentence}]->(c2)
        """
        tx.run(query,
            subject=relationship['subject'],
            object=relationship['object'],
            relationship=relationship['relationship'],
            sentence=relationship['sentence']
        )

    def create_metric_node(self, tx, metric: Dict, company: str):
        """Create metric node for financial data"""
        query = """
        MATCH (c:Company {name: $company})
        MERGE (m:Metric {value: $value, type: $type, date: $date})
        MERGE (c)-[:HAS_METRIC]->(m)
        """
        tx.run(query,
            company=company,
            value=metric['value'],
            type=metric['type'],
            date=metric['date']
        )

    def populate_graph(self, processed_data: pd.DataFrame):
        with self.driver.session() as session:
            # Create company nodes
            companies = processed_data['company'].unique()
            for company in companies:
                if company and pd.notna(company):
                    session.execute_write(self.create_company_node, company)
                    logging.info(f"Created node for company: {company}")

            # Create article nodes and relationships
            for _, article in processed_data.iterrows():
                try:
                    session.execute_write(self.create_article_node, article)
                except Exception as e:
                    logging.error(f"Error creating nodes for article: {str(e)}")



class GraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize Graph RAG with Neo4j and LangChain"""
        self.graph = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)

        # Initialize LangChain components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Groq client
        self.llm = Groq(api_key="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")
        self.vector_store = None

    def build_vector_store(self, processed_data: pd.DataFrame):
        """Build vector store from processed data"""
        documents = []
        for _, row in processed_data.iterrows():
            doc_text = f"Title: {row['title']}\nContent: {row['content']}\n"
            doc_text += f"Company: {row['company']}\n"
            doc_text += f"Sentiment: {row['sentiment']['label']}\n"
            documents.append(doc_text)

        self.vector_store = FAISS.from_texts(
            documents,
            self.embeddings,
            metadatas=[{"source": str(i)} for i in range(len(documents))]
        )

    def query_graph(self, query: str) -> Dict:
        """Query the knowledge graph using Graph RAG"""
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not initialized. Run build_vector_store first.")

            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=3)

            # Execute Cypher query
            with self.graph.driver.session() as session:
                cypher_query = self._construct_cypher_query(query)
                results = session.run(cypher_query).data()

            # Format context and results
            context = "\n".join([doc.page_content for doc in relevant_docs])
            formatted_results = json.dumps(results, indent=2)

            # Construct prompt
            prompt = f"""Please analyze the following information and answer the question:
            Question: {query}

            Context from documents:
            {context}

            Graph database results:
            {formatted_results}

            Please provide a clear and concise answer based on the above information."""

            # Use the Groq client for chat completion
            chat_completion = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing company relationships and news."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )

            response_content = chat_completion.choices[0].message.content

            return {
                'answer': response_content,
                'sources': [doc.metadata for doc in relevant_docs],
                'graph_results': results
            }

        except Exception as e:
            logging.error(f"Error in query_graph: {str(e)}")
            raise

    def _construct_cypher_query(self, question: str) -> str:
        """Construct a more detailed Cypher query based on the question"""
        if "relationship" in question.lower() or "between" in question.lower():
            return """
            MATCH (c1:Company)-[r:RELATIONSHIP]->(c2:Company)
            WHERE r.type = 'CO_MENTIONED'
            RETURN c1.name as Company1, c2.name as Company2,
                   r.type as RelationType, r.sentence as Context
            ORDER BY r.date DESC
            LIMIT 10
            """
        elif "company" in question.lower():
            return """
            MATCH (c:Company)
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(a:Article)
            RETURN c.name as Company,
                   count(a) as ArticleMentions
            ORDER BY ArticleMentions DESC
            LIMIT 5
            """
        elif "article" in question.lower():
            return """
            MATCH (a:Article)
            RETURN a.title as Title,
                   a.publishedAt as Date,
                   a.sentiment as Sentiment
            ORDER BY a.publishedAt DESC
            LIMIT 5
            """
        else:
            return """
            MATCH (n)
            RETURN DISTINCT labels(n) as NodeType,
                   count(n) as Count
            """

# main.py
def main():
    # Load environment variables
    #load_dotenv()

    try:
        with open('data\processed_news.json', 'r') as f:
            processed_data = pd.read_json(f)
    except FileNotFoundError:
        logging.error("Processed data not found. Please run Phase 2 first.")
        return

    # Initialize and populate knowledge graph
    graph_rag = GraphRAG(neo4j_uri = "bolt://44.202.210.236:7687", 
                         neo4j_user = "neo4j", 
                         neo4j_password = "sisters-car-recipients")
    graph_rag.graph.populate_graph(processed_data)

    # Build vector store
    graph_rag.build_vector_store(processed_data)

    # Example queries
    example_queries = [
        "What are the recent relationships between tech companies?"
    ]

    print("\nTesting Graph RAG with example queries:")
    print("---------------------------------------")

    for query in example_queries:
        print(f"\nQuery: {query}")
        try:
            result = graph_rag.query_graph(query)
            print(f"Answer: {result['answer']}")
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()

