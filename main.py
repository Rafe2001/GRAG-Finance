from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Add this import
from Graph_Rag.graph_rag import GraphRAG
import logging
from functools import lru_cache
from dotenv import load_dotenv
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Add CORS to allow requests from React frontend

load_dotenv()
logging.basicConfig(level=logging.INFO)

@lru_cache(maxsize=1)
def get_graph_rag():
    try:
        graph_rag = GraphRAG(
            neo4j_uri = os.getenv("NEO4J_URI"),
            neo4j_user = os.getenv("NEO4J_USER"),
            neo4j_password = os.getenv("NEO4J_PASSWORD")
        )
        
        with open('data\processed_news.json', 'r') as f:
            processed_data = pd.read_json(f)
            
        graph_rag.graph.populate_graph(processed_data)
        graph_rag.build_vector_store(processed_data)
        return graph_rag
        
    except Exception as e:
        logging.error(f"Error initializing GraphRAG: {str(e)}")
        raise

@app.route("/", methods=["POST"])
def query():
    try:
        # Get the question from the request body
        question = request.form.get("question")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Get the graph_rag instance and query it
        graph_rag = get_graph_rag()
        result = graph_rag.query_graph(question)
        
        # Return the response as JSON
        return jsonify({
            "answer": result['answer'],
            "status": "success"
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.teardown_appcontext
def cleanup(error):
    try:
        graph_rag = get_graph_rag()
        if hasattr(graph_rag, 'graph'):
            graph_rag.graph.driver.close()
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)