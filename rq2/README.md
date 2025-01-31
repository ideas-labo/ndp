# RQ2


## Installation Requirements

1.Download all the files into the same folder/clone the repository.

2.Make sure you are using Python version between 3.7 and 3.9. Otherwise, you may encounter unknown errors.

3.Using the command line: cd to the folder with the codes, and install all the required packages by running:
```
cd ./rq2
pip install -r requirements.txt
```
## Option-CPBugs Relevance Estimation

### Use a trained model
1.Use the data we reserve in the *data_pr* folder or replace it with your own data.

2.Download the [trained model](https://huggingface.co/BitMars/roberta_pf)

3.Using the command line: run the following command:

```
python roberta_ppr.py
```

You can find the rankings in the *data_pr* folder, replace the ***modelpath*** with different models to get different rankings.

### Train your own model

1.Use the data we reserve in the *data_pr* folder or replace it with your own data.

2.Download [pre-trained model](https://huggingface.co/FacebookAI/roberta-base)

3.If you use CPU training, please mask all GPU calls.

4.Using the command line: run the following command:
```
python roberta_pr_kde_gpu.py
```
You can obtain results in the terminal, compare them with our verification results, or use our results directly. Please remember to save the models that you believe perform well.

### Divide options with different densities
1.Use the data we reserve in the data_pr folder or replace with the results obtained from the model you trained yourself.

2.Using the command line: run the following command:

```
python show_rq2.py
```
You can observe the different densitiey options by GKDE in the terminal. If there are multiple delineations, please select the one that has the best performance.
