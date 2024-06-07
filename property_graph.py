import logging
import os
import sys
import time

import nest_asyncio
import streamlit as st
from icecream import ic
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.llms.openai import OpenAI
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from scr.utils import (
    check_nebula_running,
    create_storage_dirs,
    get_current_utc_datetime,
    load_config,
)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
nest_asyncio.apply()

# Load configuration
load_config(file_path="./config.yaml")
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
NEBULA_USER, NEBULA_PASSWORD, NEBULA_ADDRESS = (
    os.getenv("NEBULA_USER"),
    os.getenv("NEBULA_PASSWORD"),
    os.getenv("NEBULA_ADDRESS"),
)

Settings.llm = OpenAI(model="gpt-4o", temperature=0.3)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large",
    trust_remote_code=True,
    device="mps",
)

# Check if NebulaGraph is running
with st.sidebar:
    nebula_running = check_nebula_running()
    st.write(
        f"NebulaGraph is not running {':material/check_box:'}"
        if not nebula_running
        else "NebulaGraph is running"
    )

# Initialize Nebula connection
if "config" not in st.session_state:
    config = Config()
    config.max_connection_pool_size = 10
    st.session_state.config = config

if "connection_pool" not in st.session_state:
    connection_pool = ConnectionPool()
    connection_pool.init([("127.0.0.1", 9669)], st.session_state.config)
    st.session_state.connection_pool = connection_pool


def get_nebula_session():
    try:
        if "nebula_session" not in st.session_state:
            session = st.session_state.connection_pool.get_session(
                NEBULA_USER, NEBULA_PASSWORD
            )
            st.session_state.nebula_session = session
        return st.session_state.nebula_session
    except Exception as e:
        st.error(f"Failed to create Nebula session: {str(e)}")
        return None


def setup_nebula_graph(session):
    try:
        space_exists = session.execute("SHOW SPACES;")
        if "Publications" not in str(space_exists):
            session.execute(
                "CREATE SPACE Publications(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);"
            )
            time.sleep(10)
            session.execute("CREATE TAG entity(name string);")
            session.execute("CREATE EDGE relationship(relationship string);")
            session.execute("CREATE TAG INDEX entity_index ON entity(name(256));")

        session.execute("USE Publications;")
        st.success("NebulaGraph is set up successfully!")

    except Exception as e:
        st.error(f"Failed to setup NebulaGraph: {str(e)}")


if nebula_running:
    session = get_nebula_session()
    if session:
        setup_nebula_graph(session)

        # Check if the index already exists
        try:
            graph_store = NebulaPropertyGraphStore(
                space="Publications", overwrite=False
            )
            vec_store = SimpleVectorStore.from_persist_path(
                "./data/nebula_vec_store.json"
            )

            index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                vector_store=vec_store,
            )
            st.success("Connected to existing graph and vector store.")

        except Exception as e:
            st.error(f"Failed to connect to existing index: {str(e)}")

            # Recreate the index if it does not exist
            graph_store = NebulaPropertyGraphStore(space="Publications", overwrite=True)
            vec_store = SimpleVectorStore()

            documents = SimpleDirectoryReader(
                "scr/MDs/Big data in digital healthcare"
            ).load_data()

            index = PropertyGraphIndex.from_documents(
                documents,
                property_graph_store=graph_store,
                vector_store=vec_store,
                show_progress=True,
            )

            index.storage_context.vector_store.persist("./data/nebula_vec_store.json")
            st.success("Index created and persisted.")

        st.write(session.execute("SHOW TAGS"))
        st.write(session.execute("SHOW EDGES"))

        # Adding query and retrieval functionality
        st.header("Query and Retrieval")

        query = st.text_input("Enter your query:", "")
        if st.button("Retrieve Nodes"):
            if query:
                retriever = index.as_retriever(include_text=False)
                try:
                    nodes = retriever.retrieve(query)
                    if nodes:
                        st.write("Retrieved Nodes:")
                        for node in nodes:
                            st.write(node.text)
                    else:
                        st.write("No nodes retrieved.")
                except Exception as e:
                    st.error(f"Error during retrieval: {str(e)}")

        if st.button("Query Engine"):
            if query:
                query_engine = index.as_query_engine(include_text=True)
                try:
                    response = query_engine.query(query)
                    if response:
                        st.write("Query Response:")
                        st.write(str(response))
                    else:
                        st.write("Empty Response.")
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")


else:
    st.error("NebulaGraph is not running.")
