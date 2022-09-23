# TODO: Implement EBM Mini
from core.decision_tree_mini import DT
import numpy as np
import pandas as pd
from core.utils import rmse, make_bulk_prediction


class EBM:
    def __init__(self, num_iter=10, max_tree_size=3, learning_rate=0.01):
        self.M = num_iter
        self.max_node = max_tree_size
        self.lr = learning_rate
        self.model = None
        self._y_mu = 0

    def fit(self, X, y):
        feats_num = X.shape[1]
        self._y_mu = np.mean(y)
        models_map = {c: [] for c in range(feats_num)}
        y_target = y - self._y_mu
        for m in range(self.M):
            for feat in range(feats_num):
                model = DT(minimum_obs=1, maximum_tree_size=self.max_node)
                model = model.fit(X[:, feat].reshape(-1, 1), y_target)
                models_map[feat].append(model)
                preds = make_bulk_prediction(X[:, feat], model)
                y_target = y_target - self.lr * preds
        self.model = models_map
        return self

    def predict(self, X, return_pred_map=False):
        if self.model is None:
            raise Exception("Model not trained!")

        prediction_map = {c: 0 for c in self.model.keys()}
        for feat, models in self.model.items():
            for m in models:
                prediction_map[feat] += self.lr * DT.predict(m, X)

        return prediction_map if return_pred_map else self._y_mu + np.sum(list(prediction_map.values()))


def _test():
    df = pd.read_csv('../data/cosmetics.csv')

    # Include super feature
    df['super'] = df.Price.apply(np.log)
    _feats = ['Rank', 'Sensitive', 'super']

    train_num = int(np.ceil(df.shape[0] * 0.8))
    y = df.Price.values[0:train_num]
    y_test = df.Price.values[train_num::]
    X = df[_feats].values[0:train_num]
    X_test = df[_feats].values[train_num::]

    # Train!
    _minobs = 10
    test = EBM(num_iter=_minobs, max_tree_size=None, learning_rate=0.1)
    test = test.fit(X, y)

    pred_ebm = lambda x: test.predict(x, return_pred_map=False)
    pred_test = np.array([pred_ebm(np.asarray(X_test.reshape(-1, 1))[i, :]) for i in range(X_test.shape[0])])
    pred_train = np.array([pred_ebm(np.asarray(X.reshape(-1, 1))[i, :]) for i in range(X.shape[0])])

    print(f"TRAIN RMSE = {rmse(y, pred_train)}")
    print(f"TEST RMSE = {rmse(y_test, pred_test)}")


if __name__ == '__main__':
    _test()
