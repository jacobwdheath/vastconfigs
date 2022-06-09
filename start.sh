#!/bin/bash
apt-get install -y git
git clone https://github.com/jacobwdheath/vastconfigs.git
cd vastconfigs
mkdir model-data
mkdir datasets
cd datasets
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
cd ..
pip install -r requirements.txt


