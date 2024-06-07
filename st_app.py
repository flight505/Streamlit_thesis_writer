########### Base imports
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone
import openai

####### Streamlit imports
import streamlit as st
import streamlit.components.v1 as components

########### Bibtex imports
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

########### Icecream and loggings
from icecream import ic

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

####### Streamlit app imports
from scr.tools import PDFToMarkdownConverter, ScrapeWebsite, SearchEngine
from scr.utils import get_current_utc_datetime, load_config

################ For OpenAI

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings


# define LLM
llm = OpenAI(temperature=0, model="gpt-4o")

Settings.llm = llm
Settings.chunk_size = 512


####### Knowledge Graph with NebulaGraphStore
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore

from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()


# loads API keys from config.yaml
load_config(file_path="./config.yaml")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
pdf_folder = "scr/PDFs"
bib_folder = "scr/BIBs"
# bib_file = "scr/scr.bib"
md_folder = "scr/MDs"
chroma_db_path = "scr/scr.chroma"

CODE_KG_RAG = """

# Build Knowledge Graph with KnowledgeGraphIndex 

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

# Create a Graph RAG Query Engine

kg_rag_query_engine = kg_index.as_query_engine(
    include_text=False,
    retriever_mode="keyword",
    response_mode="tree_summarize",
)

"""


def main():
    def verify_pdf_folder(pdf_folder):
        if pdf_folder:
            os.makedirs(pdf_folder, exist_ok=True)
            pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
            if len(pdf_files) == 0:
                st.error("The folder does not contain any PDF files.")
            elif len(pdf_files) != len(os.listdir(pdf_folder)):
                st.error(
                    "The folder contains non-PDF files. Please ensure all files are PDFs."
                )

    pdf_folder = st.text_input("Enter PDF folder path:", "scr/PDFs")
    verify_pdf_folder(pdf_folder)

    if st.button("Convert PDFs to Markdown"):
        if pdf_folder:
            converter = PDFToMarkdownConverter(pdf_folder, md_folder)
            converter.convert()
        else:
            st.error("Please upload some PDF files first.")


if __name__ == "__main__":
    main()
