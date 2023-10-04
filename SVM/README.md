# Deception detection

This folder contains the main python script, models scripts and its test file along with subfolder containing test datasets. This repo is just a cleaner form of the original code taken from come880412/Deception_detection where we have removed the GUI code and make it easy for analysis for research propose.

## folder_dir
It is the folder which contains total of 30 '.jpg' images and some videos named 'd1.mp4' (deception) and 't76.mp4' (truth) and others. To test deception cases for image sequences, you need to pass the above folder as an argument in function  ```load_from_folder(folder_dir)``` which loads the first 30 images for predictions tasks. Likewise, for video, You need to append additional video name(.mp4) in the folder_dir and get respective predictions.

## lie.py
Main python script which acts as python_API to integrate it with backend. For testing for purpose, It is called from test.py file. You can read the comments for more detailed understanding.

## test.py
Used for testing purpose in deception analysis

## emotive_python_api.pdf
Some more abstract interpretation of working procedure of deception detection code.


