#!/bin/bash

mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip