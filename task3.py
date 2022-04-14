import xgboost as xgb


def loadDataset():
    # load dataset
    training_data = xgb.DMatrix(x_train, label=y_train)
    print(training_data)
    testing_data = xgb.DMatrix(x_test, label=y_test)


def loadModel():
    # model parameters
    param = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'rank:pairwise',
    }
    num_round = 10


    # train the model
    model = xgb.train(param, training_data, num_round)


    # predict
    preds = model.predict(testing_data)
    print(preds)


def evaluate_NDCG(data, k):





def main():
    loadDataset()
    loadModel()
    evaluate_NDCG()


main()