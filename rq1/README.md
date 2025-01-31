# RQ1


## Installation Requirements

1.Download all the files into the same folder/clone the repository.

2.Make sure you are using Python version between 3.7 and 3.9. Otherwise, you may encounter unknown errors.

3.Using the command line: cd to the folder with the codes, and install all the required packages by running:
```
cd ./rq1
pip install -r requirements.txt
```
## Classify CPBugs types 

### Use a trained model
1.Use the data we reserve in the *data_pp* folder or replace it with your own data.

2.Download the [trained model](https://huggingface.co/BitMars/roberta_pp)

3.Using the command line: run the following command:
```
python roberta_ppp.py
```

You can find the prediction in the data_pp folder

### Train your own model

1.Use the data we reserve in the *data_pp* folder or replace it with your own data.

2.Download [pre-trained model](https://huggingface.co/FacebookAI/roberta-base)

3.If you use CPU training, please mask all GPU calls.

3.Using the command line: run the following command:
```
python roberta_PP_KF_gpu.py
```
You can obtain results in the terminal, compare them with our verification results, or use our results directly. Please remember to save the models that you believe perform well.
