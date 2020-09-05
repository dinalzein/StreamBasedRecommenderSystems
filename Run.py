import argparse
import os
from os import path
import sys

models = {
    "KnnI": "K-Nearest Neighbors Algorithm, Item based ",
    "KnnU": "K-Nearest Neighbors Algorithm, User based ",
    "Rand": "Random Model",
    "MP": "Most Poular",
    "MF": "Matrix Factorization",
    "MFNF": "Matrix Factorization with Negative Feedback, vanilla approach",
    "MFBPR": "Matrix Factorization with Negative Feedback, BPR approach"
}

metrics = {
    "mrr": "Mean Reciprocal Rank",
    "dcg@N": "Discounted Cumulative Gain for N recommendations, i.e. dcg@5",
    "recall@N": "Recall for N recommendations, i.e. recall@5",

}


def dataset(data):
    if not path.exists("data/"+data):
        raise argparse.ArgumentTypeError("dataset file does not exist")
    return data


def train_pct(pct):
    pct = int(pct)
    if pct > 100:
        raise argparse.ArgumentTypeError("maximum training percentage is 100")
    return pct

def test_pct(pct):
    pct = int(pct)
    if pct > 100:
        raise argparse.ArgumentTypeError("maximum testing percentage is 100")
    return pct



def metric(m):
    if not any(m.lower().startswith(i) for i in ['recall@', 'dcg@']):
        if m.lower() not in ['mrr']:
            raise argparse.ArgumentTypeError("metric name is not defined")

    elif not any(map(str.isdigit, m.lower())):
        raise argparse.ArgumentTypeError("missing @N value in the metric")

    else:
        for m_ in ['recall@', 'dcg@']:
            if m_.lower() in m:
                if not m.replace(m_,'').isdecimal():
                    raise argparse.ArgumentTypeError("missing @N value in the metric")
    return m



if __name__ == "__main__":
    argvs=sys.argv[1:]
    parser = argparse.ArgumentParser("Pipeline to run the algorithms presented in this library")
    parser.add_argument(
        "-model",
        "--model_name",
        type = str,
        choices = models,
        default = "Rand",
        help = " The model that you would like to run: \n" + str(models) + ", (default: Rand)"
    )
    parser.add_argument(
        "-metric",
        "--metric",
        nargs="+",
        type = metric,
        default = ['recall@1', 'dcg@1', 'mrr'],
        help = " metrics that you would like to test" + str(metrics) + ", (default: [recall@1, dcg@1, mrr])"
    )

    parser.add_argument(
        "-dataset",
        "--dataset",
        type = dataset,
        default = "ratings100K.csv",
        help = " name of the dataset that you would like to test your algorithm on, it has to be in the data folder, (default: ratings100K.csv)"
    )

    parser.add_argument(
        "-dataset_users",
        "--dataset_users",
        default = 'None',
        help = " name of the users dataset, it has to be in the data folder, (default: None)"
    )

    parser.add_argument(
        "-dataset_items",
        "--dataset_items",
        default = 'None',
        help = " name of the items dataset, it has to be in the data folder, (default: None)"
    )


    parser.add_argument(
        "-gridsearch",
        "--gridsearch",
        type = bool,
        default = False,
        help = " if you would like to do a gridsearch to get best parameters, (default: False)"
    )

    parser.add_argument(
        "-train",
        "--train_pct",
        type = train_pct,
        default = 80,
        help = " percentage of the dataset that you would like to use for training, (default: 80)"
    )

    parser.add_argument(
        "-test",
        "--test_pct",
        type = test_pct,
        default = 20,
        help = " percentage of the dataset that you would like to use for testing, (default: 20)"
    )

    parser.add_argument(
        "-k",
        "--k",
        nargs = "+",
        default = [1],
        help = " represents the number of neighbors for the KnnI or KnnU algorithm, (default:[1, 2])"
    )

    parser.add_argument(
        "-n",
        "--n",
        nargs = "+",
        default = [1],
        help = " number of negative items you would like to consider for MF, MFNF, or MFBPR, (default:[1])"
    )

    parser.add_argument(
        "-epoch",
        "--epoch",
        nargs = "+",
        default = [1],
        help = " number of times the algorithm will work through the entire training dataset , (default:[1])"
    )

    parser.add_argument(
        "-batch",
        "--batch",
        nargs = "+",
        default = [1],
        help = " number of samples before the model is updated, (default:[1])"
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        nargs = "+",
        default = [1],
        help = " learning rate or step size, (default:[1])"
    )

    parser.add_argument(
        "-r",
        "--regularization",
        nargs="+",
        default = [1],
        help = " regularization term, (default: [1])"
    )


    args = parser.parse_args(argvs)
    #args = vars(args)
    print(args)

    ALL_FEATURES = ["model_name", "dataset", "metric", "gridsearch", "train_pct", "test_pct", "k", "n", "epoch", "batch", "learning_rate"]


    # imports
    from Dataset import Dataset
    from RunModel import RunModel
    from GridSearch import GridSearch

    #loading the dataset
    dataset = Dataset(args.dataset, str(args.dataset_users), str(args.dataset_items), args.train_pct, args.test_pct)
    train, test = dataset.get_train_test_set(args.gridsearch)
    nb_users = dataset.get_nb_users()
    nb_items = dataset.get_nb_items()

    if args.gridsearch:
        grid_search = GridSearch(args.model_name, args.k, args.n, args.epoch, args.batch, args.regularization, args.learning_rate, args.metric, nb_items, nb_users, train, test)
        grid_search.run_grid()
        pass


    else:
        run_model = RunModel(args.model_name, args.k, args.n, args.epoch, args.batch, args.regularization, args.learning_rate, args.metric, nb_items, nb_users, train, test)
        run_model.run_model()
