# For OpenAI

import logging
import os
import subprocess
import sys
import time

import streamlit as st

from scr.utils import get_current_utc_datetime, load_config

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output

import streamlit as st
from IPython.display import Markdown, display
from llama_index.core import (
    KnowledgeGraphIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.openai import OpenAI
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

# Load configuration
load_config(file_path="./config.yaml")
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
NEBULA_USER, NEBULA_PASSWORD, NEBULA_ADDRESS = (
    os.getenv("NEBULA_USER"),
    os.getenv("NEBULA_PASSWORD"),
    os.getenv("NEBULA_ADDRESS"),
)


def generate_sales_call_transcript():
    llm = OpenAI(model="gpt-4o")
    response = llm.complete(
        "Generate a sales call transcript, use real names, talk about a product, discuss some action items"
    )
    transcript = response.text
    st.write(transcript)
    logging.info(transcript)


# generate_sales_call_transcript()

# define LLM
llm = OpenAI(temperature=0, model="gpt-4o")
Settings.llm = llm
Settings.chunk_size = 512

documents = SimpleDirectoryReader("scr/MDs/Big data in digital healthcare").load_data()


def check_nebula_running():
    try:
        # Try to connect to NebulaGraph on the default port
        nebula_status = subprocess.run(
            ["nc", "-z", "127.0.0.1", "7001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return nebula_status.returncode == 0
    except Exception as e:
        logging.error(f"Failed to check NebulaGraph status: {e}")
        return False


def start_nebula():
    while not check_nebula_running():
        st.warning("NebulaGraph is not running")
    st.success("NebulaGraph has started successfully!")
    st.write(check_nebula_running())


with st.sidebar:
    st.write(
        "NebulaGraph is not running"
        if not check_nebula_running()
        else "NebulaGraph is running"
    )


config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
connection_pool.init([("127.0.0.1", 9669)], config)

session = connection_pool.get_session("root", "nebula")
if session is None:
    st.error("Failed to create session")

st.write(session.execute("SHOW SPACES"))

try:
    # Check if space exists
    space_exists = session.execute("SHOW SPACES;")
    if "paul_graham_essay" not in str(space_exists):
        # Create space if it does not exist
        session.execute(
            "CREATE SPACE paul_graham_essay(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);"
        )
        # Sleep to wait for space to be ready (simulate :sleep 10;)
        time.sleep(10)

    # Use the space
    session.execute("USE paul_graham_essay;")

    # Create schema
    session.execute("CREATE TAG IF NOT EXISTS entity(name string);")
    session.execute("CREATE EDGE IF NOT EXISTS relationship(relationship string);")
    session.execute("CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));")
    st.success("NebulaGraph is set up successfully!")
except Exception as e:
    st.error(f"Failed to setup NebulaGraph: {str(e)}")
finally:
    connection_pool.close()


space_name = "paul_graham_essay"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

graph_store.query("SHOW HOSTS")

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

query_engine = index.as_query_engine()


response = query_engine.query("Tell me more about Interleaf")


st.markdown(response)

# WikipediaReader = download_loader("WikipediaReader")
# loader = WikipediaReader()
# documents = loader.load_data(pages=["Advanced Encryption Standard"], auto_suggest=False)

# kg_index = KnowledgeGraphIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     service_context=service_context,
#     max_triplets_per_chunk=10,
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
# )
