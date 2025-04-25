#!/bin/bash

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential

# Install NEST simulator
pip3 install nest-asyncio

# Clone the project repository
git clone https://github.com/ErmannoF00/Embedded_SNN.git
cd Embedded_SNN

# Run the simulation
python3 simulate.py
