import os
from datetime import datetime, timezone

import bibtexparser
import chromadb
import streamlit as st
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

# from chromadb.utils import embedding_function
from scr.prompts import (
    manager_description,
    manager_instructions,
    mission_statement_prompt,
    researcher_description,
    researcher_instructions,
)
from scr.tools import ScrapeWebsite, SearchEngine
from scr.utils import load_config

# Set the environment variable
os.environ["IN_STREAMLIT"] = "true"

# loads API keys from config.yaml
load_config(file_path="./config.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define paths
pdf_folder = "scr/PDFs"
bib_file = "scr/scr.bib"
md_folder = "scr/MDs"
chroma_db_path = "scr/scr.chroma"


def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


# Function to convert all PDFs in a folder to markdown using Marker
def pdf_to_md(input_folder, output_folder):
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        command = f"marker_single {pdf_path} {output_folder} --batch_multiplier 12 --langs English"
        os.system(command)
    os.system(command)


# Function to create embeddings and ChromaDB
def create_chroma_db(md_folder, bib_file, db_path):
    client = chromadb.Client()
    collection = client.create_collection("academic_papers")

    # Read and parse .bib file
    with open(bib_file) as bibtex_file:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibtex_file, parser=parser)

    for md_file in os.listdir(md_folder):
        if md_file.endswith(".md"):
            with open(os.path.join(md_folder, md_file), "r") as f:
                text = f.read()
            emb = embedding_function(text)
            collection.add({"content": text, "embedding": emb})

    client.save(db_path)


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
            else:
                st.success(
                    "Folder verified! All files are PDFs and at least one PDF exists."
                )
            st.success("pdf folder successfully set!")

    pdf_folder = st.text_input("Enter PDF folder path:", "scr/PDFs")
    verify_pdf_folder(pdf_folder)

    if st.button("Convert PDFs to Markdown"):
        if pdf_folder:
            pdf_to_md(pdf_folder, md_folder)
            st.success("Conversion completed!")
        else:
            st.error("Please upload some PDF files first.")

    manager_instructions_with_datetime = manager_instructions.format(
        datetime=get_current_utc_datetime()
    )

    st.write(manager_instructions_with_datetime)


if __name__ == "__main__":
    main()
