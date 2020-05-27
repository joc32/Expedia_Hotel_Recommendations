# Expedia Hotel Recommendations

This project is an implementation of Kaggle Expedia Hotel Recommendation [competition](https://www.kaggle.com/c/expedia-hotel-recommendations/overview) based on the hybrid approach proposed by Wang, Sun and Lin in [Hotel Recommendation Based on Hybrid Model](http://cs229.stanford.edu/proj2016spr/report/041.pdf). The approach develops model based on combination of collaborative and content-based filtering.

## Datasets
Before start, download the datasets from Kaggle and import them to `/datasets` folder in following format:
* `/datasets/train.csv`
* `/datasets/test.csv`

Model can be alternatively run on `/datasets/1percent.csv` which contains 1% subset of original `train.csv` dataset.

## Installation
Model was developed with Python 3.7 version.
Use package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies
```shell script
pip3 install -r requirements.txt
```

## Run

Run main python script to calculate recommendations. Output will be generated in the format of Kaggle submission file (Map@5) to `/datasets/test_results.txt`.
```shell script
python3 main.py
```

You can also specify slice on which to make a recommendation (for development purposes). Run script with percentage as an argument
```shell script
python3 main.py 0.05
```
Run recommendations on 5% subset of dataset.

