# Financial News Knowledge Graph

This project builds a knowledge graph from financial news articles and uses a Graph RAG (Retrieval Augmented Generation) pipeline to answer questions about company relationships and financial events.
![WhatsApp Image 2025-01-17 at 17 56 02_b2433a0d](https://github.com/user-attachments/assets/1a40499a-b254-488d-b01a-89963c2ca4fc)


## Project Phases

The project is divided into three phases:

### Phase 1: Data Ingestion

- **Objective:** Collect relevant financial news articles from reliable sources.
- **Implementation:** 
    - Uses the News API to fetch articles related to a list of target companies and keywords.
    - Scrapes additional data if needed.
    - Saves the collected data in a structured format (e.g., CSV).
- **Key Components:**
    - `data_collection.py`: Script for collecting news data using News API or web scraping.
    - `data/raw_news.csv`: Raw news articles collected.
    - `data/tech_news.csv`: CSV containing cleaned data of tech news.
    - `tech_companies`: A list of target company names.


### Phase 2: NLP Pipeline

- **Objective:** Extract entities, relationships, and sentiment from news articles using NLP techniques.
- **Implementation:** 
    - Utilizes spaCy for Named Entity Recognition (NER).
    - Employs custom patterns or dependency parsing to identify relationships between entities.
    - Performs sentiment analysis using transformers (e.g., FinBERT).
    - Stores the extracted information in JSON format.
- **Key Components:**
    - `nlp_pipeline.py`: Script for NLP processing (NER, relationship extraction, sentiment analysis).
    - `data/processed_news.json`: JSON file containing extracted entities, relationships, and sentiment for each article.


### Phase 3: Graph RAG Pipeline

- **Objective:** Build a knowledge graph and integrate it with a RAG pipeline for question answering.
- **Implementation:**
    - Creates a knowledge graph in Neo4j using the extracted entities and relationships.
    - Builds a vector store (e.g., FAISS) to index the news articles for retrieval.
    - Implements a Graph RAG pipeline that retrieves relevant documents and queries the knowledge graph using LangChain.
    - Generates answers to user questions based on the retrieved information.
- **Key Components:**
    - `graph_rag_pipeline.py`: Script for knowledge graph creation and Graph RAG pipeline implementation.
    - `neo4j_credentials.env`: Neo4j connection details (stored securely, not in the repository).
    - `groq_credentials.env`: Groq API key (stored securely, not in the repository).

## Usage

1.  **Data Ingestion:**
bash python data_collection.py
 
2.  **NLP Processing:**
bash python nlp_pipeline.py

3.  **Graph RAG Pipeline:**
bash python graph_rag_pipeline.py


## Contributing

Feel free to contribute to the project by:

-   Reporting bugs or suggesting improvements.
-   Adding new features or enhancing existing ones.
-   Improving documentation or testing.

## License

[Specify your project's license here]
