# Financial News Knowledge Graph

This project builds a **Knowledge Graph** from financial news articles and leverages a **Graph RAG (Retrieval Augmented Generation)** pipeline to answer questions about company relationships and financial events. The knowledge graph integrates extracted entities, relationships, and sentiment analysis to provide comprehensive and intelligent answers to user queries.

![WhatsApp Image 2025-01-17 at 17 56 02_b2433a0d](https://github.com/user-attachments/assets/1a40499a-b254-488d-b01a-89963c2ca4fc)

### Live App: [Explore the App](https://grag-finance-2.onrender.com)

---

## Table of Contents

- [Project Phases](#project-phases)
- [App Overview](#app-overview)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Contributing](#contributing)
- [Technology Stack](#technology-stack)
- [Future Enhancements](#future-enhancements)

---

## Project Phases

The project is divided into three key phases:

### Phase 1: Data Ingestion

- **Objective:** Collect relevant financial news articles from trusted sources.
- **Implementation:**
    - Uses the **News API** to fetch articles related to target companies and keywords.
    - Optionally scrapes additional data if needed.
    - Saves the collected data in a structured format (e.g., CSV).
- **Key Components:**
    - `Ingestion/data_ingestion.py`: Script for collecting news data using the News API or web scraping.
    - `data/raw_news.csv`: Raw collected news articles.
    - `data/tech_news.csv`: Cleaned and processed data of tech-related news.
    - `tech_companies.txt`: A list of target company names.

---

### Phase 2: NLP Pipeline

- **Objective:** Extract structured entities, relationships, and sentiment from news articles.
- **Implementation:**
    - Uses **spaCy** for Named Entity Recognition (NER).
    - Applies custom patterns or dependency parsing to extract relationships between entities.
    - Sentiment analysis is performed using a **FinBERT** model.
    - Extracted data is stored in **JSON format**.
- **Key Components:**
    - `processing/process.py`: Script for performing NLP tasks, including NER, relationship extraction, and sentiment analysis.
    - `data/processed_news.json`: JSON file containing the extracted entities, relationships, and sentiment for each article.

---

### Phase 3: Graph RAG Pipeline

- **Objective:** Build a knowledge graph and integrate it with a RAG pipeline for question answering.
- **Implementation:**
    - Creates a knowledge graph in **Neo4j** using the extracted entities and relationships.
    - Builds a vector store (e.g., **FAISS**) for efficient document retrieval.
    - Implements a **Graph RAG pipeline** using **LangChain**, combining document retrieval and graph querying for answering complex questions.
    - **Groq API** is used for text generation and reasoning.
- **Key Components:**
    - `graph_rag_pipeline.py`: Script for knowledge graph creation and the Graph RAG pipeline implementation.
    - `neo4j_credentials.env`: Neo4j connection details (stored securely).
    - `groq_credentials.env`: Groq API key for LLM capabilities (stored securely).

---

## App Overview

This project also includes a **Flask-based web application** that allows users to interact with the Financial News Knowledge Graph via a simple interface. Users can query the knowledge graph to explore entities, relationships, and get detailed answers to questions about financial events and company interactions.

- **Key Features of the App:**
    - **Search Interface:** Allows users to enter queries about companies, financial events, or other relevant entities.
    - **Answer Generation:** Combines information from the knowledge graph and retrieved documents to provide accurate answers.
    - **Dynamic Interaction:** Built using **Flask**, the web interface is simple, fast, and interactive for end-users.

---

## Usage

To run the project locally or deploy it, follow these steps:

1. **Data Ingestion:**
    ```bash
    python Ingestion/data_ingestion.py
    ```

2. **NLP Processing:**
    ```bash
    python processing/process.py
    ```

3. **Graph RAG Pipeline:**
    ```bash
    python Graph_Rag/graph_rag.py
    ```

4. **Run Flask App:**
    To launch the Flask web application, run:
    ```bash
    python main.py
    ```

   This will start a local server. You can visit the app by navigating to `http://localhost:5000` in your browser.

---

## Contributing

Contributions to the project are welcome! Here are some ways you can help:

- Reporting bugs or suggesting improvements.
- Adding new features or enhancing existing ones.
- Improving documentation or testing.

---

## Technology Stack

- **Programming Languages:** Python
- **News API:** For fetching real-time financial news articles.
- **spaCy:** For Named Entity Recognition (NER).
- **Transformers:** For sentiment analysis (e.g., FinBERT).
- **Neo4j:** For knowledge graph storage and querying.
- **FAISS:** For indexing and retrieving documents.
- **LangChain:** For building the Graph RAG pipeline.
- **Groq:** For text generation and reasoning via LLM.
- **Flask:** For the web application backend.
- **HTML, CSS (Bootstrap), JS (jQuery):** For building a responsive and interactive front-end interface.

---

## Future Enhancements

- **Improved Relationship Extraction:** Explore more advanced NLP techniques to extract complex relationships from articles.
- **Real-time Updates:** Implement automatic updates to the knowledge graph with new articles.
- **Visualization Tools:** Develop tools to visualize the knowledge graph and better explore the relationships.
- **Enhanced User Interface:** Improve the user interface to offer a more intuitive and user-friendly experience for querying and interaction.
- **Domain-Specific Knowledge:** Integrate specialized financial domain knowledge to enhance the depth and accuracy of analysis.

---

This README provides an overview of the **Financial News Knowledge Graph** project. For more detailed instructions, refer to the individual scripts and their comments. If you have any questions, feel free to reach out or contribute to the project!
