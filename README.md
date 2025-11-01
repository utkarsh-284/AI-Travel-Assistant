# AI Travel Assistent for Vietnam

This project implements a hybrid AI travel Assistant that provides personalized travel recommendations for Vietnam. It leverages a combination of semantic search using Pinecone and a knowledge graph using Neo4j to deliver comprehensive and contextually relevant travel itineraries. The AI components are powered by NVIDIA's API for generating embeddings and chat responses.

## Project Overview

The AI travel Assistant is designed to understand a user's travel query, retrieve relevant information from a vector database and a graph database, and then generate a detailed travel plan. The project is structured as a series of Python scripts that handle data processing, database population, and the main chat application.

### Key Features

- **Hybrid Search:** Combines semantic search for finding relevant entities and graph-based search for discovering connections and relationships between them.
- **RAG Pipeline:** Implements a Retrieval-Augmented Generation (RAG) pipeline to provide grounded and informative responses.
- **Stateful Agent:** Uses LangGraph to create a stateful and modular agent, making the logic easy to follow, maintain, and extend.
- **NVIDIA Integration:** Utilizes NVIDIA's API for high-quality embeddings and chat models.
- **Data Visualization:** Includes a script to visualize the knowledge graph stored in Neo4j.

## Getting Started

To get the AI travel Assistant up and running, you will need to set up the required services, configure your API keys, and run the data loading scripts.

### Prerequisites

- Python 3.10+
- Docker
- An active internet connection

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys and Configuration

You will need API keys from NVIDIA and Pinecone, as well as credentials for your Neo4j database.

1.  **NVIDIA API Key:**
    -   Obtain your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys).

2.  **Pinecone API Key:**
    -   Log in to your [Pinecone account](https://www.pinecone.io/) to get your API key.

3.  **Neo4j Database:**
    -   Run a Neo4j container using the following Docker command. This will start a Neo4j instance with the necessary ports exposed and authentication set to `neo4j/password`.

    ```bash
    docker run --restart always --publish=7474:7474 --publish=7687:7687 --env NEO4J_AUTH=neo4j/password neo4j:2025.10.1
    ```

4.  **Configure the Project:**
    -   Create a `config.py` file in the root of the project.
    -   Add your API keys and database credentials to the `config.py` file, using `config_example.py` as a template.

### 4. Data Loading

Before you can start chatting with the assitant, you need to load the travel data into Pinecone and Neo4j.

1.  **Upload to Pinecone:**
    -   Run the `pinecone_upload.py` script to generate embeddings for the dataset and upload them to your Pinecone index.

    ```bash
    python pinecone_upload_nvidia.py
    ```

2.  **Load into Neo4j:**
    -   After starting the Neo4j Docker container, run the `load_to_neo4j.py` script to populate the graph database.

    ```bash
    python load_to_neo4j.py
    ```

### 5. Running the Chat Assistant

Once the data is loaded, you can start the AI travel Assistant:

```bash
python hybrid_chat_langgraph.py
```

The script will prompt you to enter your travel questions. Type `exit` to quit the application.

### (Optional) Visualizing the Graph

To visualize the knowledge graph, you can run the `visualize_graph.py` script. This will generate an HTML file (`neo4j_viz.html`) that you can open in your browser.

```bash
python visualize_graph.py
```

## Project Structure

-   `config.py`: Configuration file for API keys and database credentials.
-   `hybrid_chat_langgraph.py`: The main chat application that runs the RAG pipeline.
-   `improvements.md`: A document detailing the architectural and performance enhancements.
-   `load_to_neo4j.py`: Script to load the travel dataset into the Neo4j graph database.
-   `pinecone_upload_nvidia.py`: Script to upload the dataset with embeddings to Pinecone.
-   `requirements.txt`: A list of all the Python dependencies.
-   `vietnam_travel_dataset.json`: The raw travel data in JSON format.
-   `visualize_graph.py`: Script to generate a visualization of the Neo4j graph.

## 📬 Contact
**Utkarsh Bhardwaj**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Utkarsh284-blue)](https://www.linkedin.com/in/utkarsh284/)
[![GitHub](https://img.shields.io/badge/GitHub-utkarsh--284-lightgrey)](https://github.com/utkarsh-284)

**Contact:** ubhardwaj284@gmail.com

**Publish Date:** 1st November, 2025