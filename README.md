# Stream Based Recommender Systems

__This repository aims at implementing different algorithms for stream based recommender systems including:__
1. [Most Popular (MP)](MP.py)
2. [K-Nearest Neighbors, User Based (KNNU)](KnnU.py)
3. [K-Nearest Neighbors, Item Based (KNNI)](KnnI.py)
4. [Matrix Factorization (MF)](MF.py)
5. [Matrix Factorization with Negative Feedback (MFNF)](MFNF.py)
6. [Bayesian Personalized Ranking for Matrix Factorization (MFBPR)](MFBPR.py)
7. [Random](Random.py)

__For evaluating of the models, we implement the following metrics:__
1. Mean Reciprocal Rank (MRR)
2. recall@N
3. DCG@N

## Setup environment

To install the dependencies, run the following in the folder where [requirements.txt](requirements.txt) is stored:
```Bash
pip install -r requirements.txt
```


## Datasets
We will be using the [MovieLens datasets](https://grouplens.org/datasets/movielens/1m/) containing 1000209 interactions with 6040 users and 3706 items.

## Evaluating the models
To evaluate the model, you need to pass some arguments in order to run correctly the script; to check these arguments run:
```Bash
python Run.py --help
```
Otherwise, you can run the script with the default arguments:
```Bash
python Run.py
```

A [report](./report.pdf) has been made to give an overview of the papers cited in and analyze the results of the experiments.


## Results for different algorithms
For all results, larger is better.
The results below present the evaluation of all the implemented algorithms for recall@N, DCG@N, and MRR metrics respectively. For neighborhood-based approaches, we averaged the results for K = [5, 10, 20, 50, 100]

|                    | Random  |  MP     | MF      | MFNF    | KNNU    | KNNI
|---                 |---      |---      |---      |---      |---      |---     
Recall@1             | 0.00038 | 0.00875 | 0.00387 | 0.00221 | 0.00623 | 0.00623
Recall@5             | 0.00170 | 0.03586 | 0.01714 | 0.05868 | 0.01407 | 0.01407
Recall@10            | 0.00306 | 0.06737 | 0.03276 | 0.10701 | 0.02519 | 0.02519
Recall@50            | 0.01579 | 0.21800 | 0.13212 | 0.28468 | 0.09258 | 0.11363
Recall@100           | 0.031577| 0.33207 | 0.23051 | 0.37954 | 0.18685 | 0.21233


|                    | Random  |  MP     | MF      | MFNF    | KNNU    | KNNI
|---                 |---      |---      |---      |---      |---      |---     
DCG@1                | 0.00038 | 0.00087 | 0.0043  | 0.00256 | 0.00397 | 0.00024
DCG@5                | 0.00170 | 0.00365 | 0.01951 | 0.05767 | 0.01016 | 0.00187
DCG@10               | 0.00307 | 0.00694 | 0.03689 | 0.11574 | 0.01728 | 0.00347
DCG@50               | 0.01588 | 0.02385 | 0.15113 | 0.42441 | 0.06184 | 0.02261
DCG@100              | 0.03194 | 0.03786 | 0.26694 | 0.66884 | 0.06906 | 0.04359


|                    | Random  |  MP     | MF      | MFNF    | KNNU    | KNNI
|---                 |---      |---      |---      |---      |---      |---     
MRR                  | 0.00278 | 0.03206 | 0.01793 | 0.03819 | 0.01596 | 0.01637


Note: This code is done 2 years ago. 
