#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Uninstall problematic packages
pip uninstall -y jax jaxlib

# Run the Streamlit app
streamlit run major_project.py
