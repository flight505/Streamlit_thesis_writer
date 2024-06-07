import json
import os
import time
from datetime import datetime, timezone

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pydantic import Field

from scr.utils import get_current_utc_datetime, load_config

load_config(file_path="./config.yaml")


class SearchEngine:
    """
    SearchEngine: A search engine tool. You can use this tool to search for a specific query on a search engine.
    The output of the search engine is a dictionary where the key is the source of the information and the value is the content.
    """

    search_engine_query: str = Field(
        ..., description="Search engine query to be executed by the tool"
    )

    def format_results(self, organic_results):
        result_strings = []
        for result in organic_results:
            title = result.get("title", "No Title")
            link = result.get("link", "#")
            snippet = result.get("snippet", "No snippet available.")
            result_strings.append(
                f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---"
            )

        return "\n".join(result_strings)

    def run(self):
        search_url = "https://google.serper.dev/search"
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": os.environ[
                "SERPER_DEV_API_KEY"
            ],  # Ensure this environment variable is set with your API key
        }
        payload = json.dumps({"q": self.search_engine_query})

        # Attempt to make the HTTP POST request
        try:
            response = requests.post(search_url, headers=headers, data=payload)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4XX, 5XX)
            results = response.json()

            # Check if 'organic' results are in the response
            if "organic" in results:
                return self.format_results(results["organic"])
            else:
                return "No organic results found."

        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"
        except requests.exceptions.RequestException as req_err:
            return f"Request exception occurred: {req_err}"
        except KeyError as key_err:
            return f"Key error in handling response: {key_err}"


class ScrapeWebsite:
    """
    ScrapeWebsite: A website scraping tool. You can use this tool to scrape the content of a website.
    You must provide the URL of the website you want to scrape.
    """

    website_url: str = Field(
        ..., description="The URL of the website to scrape the content from."
    )

    def run(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Accept-Encoding": "gzip, deflate, br",
        }

        try:
            # Making a GET request to the website
            response = requests.get(self.website_url, headers=headers, timeout=15)
            response.raise_for_status()  # This will raise an exception for HTTP errors

            # Parsing the page content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n")
            # Cleaning up the text: removing excess whitespace
            clean_text = "\n".join(
                [line.strip() for line in text.splitlines() if line.strip()]
            )

            print(f"Successfully scraped content from {self.website_url}")

            return {self.website_url: clean_text}

        except requests.exceptions.RequestException as e:
            print(f"Error retrieving content from {self.website_url}: {e}")
            return {
                self.website_url: f"Failed to retrieve content due to an error: {e}"
            }


class PDFToMarkdownConverter:
    def __init__(self, input_folder, output_folder, log_file="converted_files.json"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.log_file = log_file

    def convert(self):
        self.ensure_directory_exists(self.input_folder)
        self.ensure_directory_exists(self.output_folder)

        converted_files_log = self.load_converted_files_log()
        pdf_files = [f for f in os.listdir(self.input_folder) if f.endswith(".pdf")]

        total_files = len(pdf_files)
        already_converted = sum(
            1
            for f in pdf_files
            if os.path.splitext(f)[0] + ".md" in converted_files_log
        )
        to_convert = total_files - already_converted

        with st.spinner("Converting PDF files to Markdown..."):
            msg = st.toast("Checking for PDF files to convert...", icon="🔎")
            time.sleep(0.5)
            if not pdf_files:
                msg.toast("No PDF files found in the input folder.", icon="👎")
                st.stop()
            if to_convert == 0:
                msg.toast("All PDF files are already converted.", icon="👍")
            else:
                msg.toast(
                    f"Total: {total_files}, Already Converted: {already_converted}, To Convert: {to_convert}"
                )

            converted_files_count = 0
            for i, pdf_file in enumerate(pdf_files):
                if not self.convert_pdf_to_md(pdf_file, converted_files_log):
                    continue
                converted_files_count += 1
            if not to_convert == 0:
                time.sleep(0.5)
                st.toast(f"Converting {i+1}/{total_files} files...", icon="☕")

            self.save_converted_files_log(converted_files_log)
            time.sleep(0.5)
            if msg is not None:
                msg.toast(
                    f"Conversion completed! {total_files} are ready to be used.",
                    icon="👍",
                )
            else:
                st.toast(
                    f"Conversion completed! {total_files} are ready to be used.",
                    icon="👍",
                )

    def ensure_directory_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def load_converted_files_log(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                return json.load(f)
        return []

    def convert_pdf_to_md(self, pdf_file, converted_files_log):
        pdf_path = os.path.join(self.input_folder, pdf_file)
        md_file = os.path.splitext(pdf_file)[0] + ".md"
        md_path = os.path.join(self.output_folder, md_file)

        if md_file in converted_files_log:
            return False

        command = f"marker_single '{pdf_path}' '{md_path}' --batch_multiplier 12 --langs English"
        os.system(command)
        converted_files_log.append(md_file)
        return True

    def save_converted_files_log(self, converted_files_log):
        with open(self.log_file, "w") as f:
            json.dump(converted_files_log, f)
