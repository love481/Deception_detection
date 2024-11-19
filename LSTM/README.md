# LSTM based model
We use peak Action unit labels triggered from sequential 60 frames from each video segment using [code](https://github.com/lingjivoo/OpenGraphAU) and utilize the LSTM model for prediction tasks(deception or not). The sample.json file contains the action unit encoding values and corresponding gt labels(deception or not) computed for real-life trial videos.

## Running training and testing script
`` python lstm_model.py ``

## Results
The peak test accuracy obtained is 79.12% for testing samples from LSTM based deception detector.

