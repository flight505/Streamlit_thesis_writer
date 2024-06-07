import logging
import os
import subprocess
from datetime import datetime, timezone

import streamlit as st
import yaml
from llama_index.core.schema import QueryBundle


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            os.environ[key] = value


def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


def check_nebula_running():
    try:
        result = subprocess.run(
            ["nc", "-z", "127.0.0.1", "9669"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Failed to check NebulaGraph status: {e}")
        return False


def create_storage_dirs(paths):
    for directory in paths:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


class DatabaseInspector:
    """Inspect the NebulaGraph and the Vector database.
    - Usage
    inspector = DatabaseInspector(connection_pool, vector_retriever)
    inspector.inspect_nebula_graph()
    inspector.inspect_vector_database()
    """

    def __init__(self, connection_pool, vector_retriever):
        self.connection_pool = connection_pool
        self.vector_retriever = vector_retriever

    def inspect_nebula_graph(self):
        session = None
        try:
            session = self.connection_pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
            if session is None:
                st.error("Failed to create session")
            else:
                # Check if the space exists
                space_exists = session.execute("SHOW SPACES;")
                st.write("Spaces:", space_exists)

                # Use the space and check tags and edges
                session.execute("USE Publications;")
                tags = session.execute("SHOW TAGS;")
                edges = session.execute("SHOW EDGES;")
                st.write("Tags:", tags)
                st.write("Edges:", edges)
        except Exception as e:
            st.error(f"Failed to inspect NebulaGraph: {str(e)}")
        finally:
            if session:
                session.release()
            self.connection_pool.close()

    def inspect_vector_database(self):
        try:
            # Retrieve some documents from the vector index
            query_bundle = QueryBundle(query="Retrieve some documents")
            retrieved_nodes = self.vector_retriever.retrieve(query_bundle)

            # Display the retrieved documents and their embeddings
            for node in retrieved_nodes:
                st.write(f"Node ID: {node.node.node_id}")
                st.write(f"Text: {node.node.text}")
                st.write(f"Embedding: {node.node.embedding}")
        except Exception as e:
            st.error(f"Failed to inspect vector database: {str(e)}")
