#!/bin/bash
apt update -y
apt-get install -y zip
apt-get install -y git
git clone https://github.com/jacobwdheath/vastconfigs.git
cd vastconfigs
mkdir model-data
mkdir datasets
cd datasets
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
cd vastconfigs
pip install -r requirements.txt


