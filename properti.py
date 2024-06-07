import logging
import os
import subprocess
import sys
import time
from typing import List

import streamlit as st
from icecream import ic

# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core import (
    KnowledgeGraphIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    BaseRetriever,
    KeywordTableSimpleRetriever,
    KGTableRetriever,
    RecursiveRetriever,
    VectorIndexRetriever,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.openai import OpenAI
from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

from scr.utils import (
    check_nebula_running,
    create_storage_dirs,
    get_current_utc_datetime,
    load_config,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load configuration
load_config(file_path="./config.yaml")
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
NEBULA_USER, NEBULA_PASSWORD, NEBULA_ADDRESS = (
    os.getenv("NEBULA_USER"),
    os.getenv("NEBULA_PASSWORD"),
    os.getenv("NEBULA_ADDRESS"),
)


# Configure global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)
Settings.chunk_size = 512


with st.sidebar:
    st.write(
        f"NebulaGraph is not running {':material/check_box:'}"
        if not check_nebula_running()
        else "NebulaGraph is running"
    )

config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
connection_pool.init([("127.0.0.1", 9669)], config)

session = connection_pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
if session is None:
    st.error("Failed to create session")
else:
    st.write(session.execute("SHOW SPACES"))

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
    st.write(session.execute("SHOW SPACES"))


except Exception as e:
    st.error(f"Failed to setup NebulaGraph: {str(e)}")
finally:
    if session:
        session.release()
    connection_pool.close()

space_name = "Publications"
edge_types, rel_prop_names = [], []
tags = []

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

create_storage_dirs(["./scr/storage_graph", "./scr/storage_vector"])
# Create storage contexts for persistence
storage_context = StorageContext.from_defaults(
    # persist_dir="./scr/storage_graph",
    graph_store=graph_store
)
storage_context_vector = StorageContext.from_defaults(
    # persist_dir="./scr/storage_vector"
)
from llama_index import load_index_from_storage

storage_context_vector = StorageContext.from_defaults(persist_dir="./storage_vector")
vector_index = load_index_from_storage(
    service_context=service_context, storage_context=storage_context_vector
)

try:
    documents = SimpleDirectoryReader(
        "scr/MDs/Big data in digital healthcare"
    ).load_data()

    embedding = OpenAIEmbedding()

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
        include_embeddings=True,
        verbose=True,
    )

    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context_vector
    )
    vector_retriever = VectorIndexRetriever(vector_index)

    # Implement CustomRetriever
    kg_retriever = KGTableRetriever(kg_index)

    # Persist the storage contexts
    storage_context.persist(persist_dir="./scr/storage_graph")
    storage_context_vector.persist(persist_dir="./scr/storage_vector")


except Exception as e:
    logging.error(f"Error occurred while creating indices or retrievers: {str(e)}")
    st.error(f"Error occurred while creating indices or retrievers: {str(e)}")
    raise

st.write(ic(kg_index))


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


custom_retriever = CustomRetriever(vector_retriever, kg_retriever)


# # create custom retriever
# vector_retriever = VectorIndexRetriever(index=vector_index)
# kg_retriever = KGTableRetriever(
#     index=kg_index, retriever_mode="keyword", include_text=False
# )
# Inspect the Knowledge Graph (KG) database


# Usage
inspector = DatabaseInspector(connection_pool, vector_retriever)
inspector.inspect_nebula_graph()
inspector.inspect_vector_database()


# create response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

graph_vector_rag_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

vector_query_engine = vector_index.as_query_engine()


st.stop()


#     kg_keyword_query_engine = kg_index.as_query_engine(
#         # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
#         include_text=False,
#         retriever_mode="keyword",
#         response_mode="tree_summarize",
#     )
#     response = kg_keyword_query_engine.query(
#         "Explain the initiatives mentioned in Big data in digital healthcare, and give me a table with the citations in the papers related to it"
#     )

#     st.markdown(response)
# except Exception as e:
#     st.error(f"Failed to create Knowledge Graph Index or Query: {str(e)}")
