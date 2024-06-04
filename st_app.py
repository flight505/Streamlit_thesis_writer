import os
from datetime import datetime, timezone
import streamlit as st

from prompts import (
    manager_description,
    manager_instructions,
    mission_statement_prompt,
    researcher_description,
    researcher_instructions,
)
from tools import ScrapeWebsite, SearchEngine
from utils import load_config

# loads API keys from config.yaml
load_config(file_path="./config.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_current_utc_datetime():
    now_utc = datetime.now(timezone.utc)
    current_time_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    return current_time_utc


def main():
    manager_instructions_with_datetime = manager_instructions.format(
        datetime=get_current_utc_datetime()
    )

    st.write(manager_instructions_with_datetime)


if __name__ == "__main__":
    main()
