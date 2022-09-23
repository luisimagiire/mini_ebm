from core.decision_tree_mini import DT
import numpy as np
import pandas as pd
from core.utils import rmse, make_bulk_prediction
from tqdm import tqdm
import itertools
from typing import Optional


class EBM:
    def __init__(self, num_iter: int = 10, max_tree_size: Optional[int] = 3, learning_rate: float = 0.01):
        self.M = num_iter
        self.max_node = max_tree_size
        self.lr = learning_rate
        self.model = None
        self._y_mu = 0

    def fit(self, X, y):
        feats_num = [str(c) for c in range(X.shape[1])]
        self._y_mu = np.mean(y)
        self.model = {c: [] for c in feats_num}
        y_target = y - self._y_mu
        for m in tqdm(range(self.M)):
            for feat in feats_num:
                _m_feat = int(feat) if "_" not in feat else [int(c) for c in feat.split('_')]
                model = DT(minimum_obs=1, maximum_tree_size=self.max_node)
                if isinstance(_m_feat, int):
                    model = model.fit(X[:, _m_feat].reshape(-1, 1), y_target)
                else:
                    model = model.fit(X[:, _m_feat], y_target)
                preds = make_bulk_prediction(X[:, _m_feat], model)
                model_predictions = self.predict_bulk(X)
                y_target = y - (model_predictions + self.lr * preds)
                self.model[feat].append(model)

            if m == 0:
                # Find pair-wise interactions
                # Original EBM uses FAST algorithm to select suitable pair-wise interaction
                # Out of scope for us since that is mainly to control the computational cost of fitting pair-wise models
                feature_ss = list(itertools.combinations(feats_num, 2))
                model_predictions = self.predict_bulk(X)
                base_rss = np.sum(y - model_predictions)
                for pair in feature_ss:
                    _idx_pair = [idx for idx in map(int, pair)]
                    model = DT(minimum_obs=1, maximum_tree_size=self.max_node)
                    model = model.fit(X[:, _idx_pair], y_target)
                    preds = make_bulk_prediction(X[:, _idx_pair], model)
                    iter_resids = y - (model_predictions + self.lr * preds)
                    if np.sum(iter_resids) < base_rss:
                        m_tag = "_".join(pair)
                        self.model[m_tag] = [model]
                        feats_num.append(m_tag)

            print(f"RUN {m} - RMSE {rmse(y, self.predict_bulk(X))}")
        return self

    def predict_bulk(self, X):
        return np.array([self.predict(np.asarray(X)[i, :]) for i in range(X.shape[0])])

    def predict(self, X, return_pred_map=False):
        if self.model is None:
            raise Exception("Model not trained!")

        prediction_map = {c: 0 for c in self.model.keys()}
        for feat, models in self.model.items():
            for m in models:
                if "_" not in feat:
                    prediction_map[feat] += self.lr * DT.predict(m, X.reshape(-1, 1)[int(feat)])
                else:
                    prediction_map[feat] += self.lr * DT.predict(m, X[[int(c) for c in feat.split("_")]])

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
    _minobs = 5
    test = EBM(num_iter=_minobs, max_tree_size=None, learning_rate=0.5)
    test = test.fit(X, y)

    pred_test = test.predict_bulk(X_test)
    pred_train = test.predict_bulk(X)

    print(f"TRAIN RMSE = {rmse(y, pred_train)}")
    print(f"TEST RMSE = {rmse(y_test, pred_test)}")


if __name__ == '__main__':
    _test()
