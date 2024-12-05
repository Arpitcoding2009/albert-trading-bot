#!/bin/bash

# Update package lists
apt-get update

# Install build-essential and TA-Lib dependencies
apt-get install -y build-essential libta-lib0-dev

# Install Python dependencies
pip install -r requirements.txt
