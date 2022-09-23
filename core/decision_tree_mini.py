# Create a decision tree class
import numpy as np
import pandas as pd
from core.utils import rmse, make_bulk_prediction

class DT:
    def __init__(self, value=None, ln=None, rn=None, minimum_obs=20, maximum_tree_size=None, current_height=0):
        self.ln = None
        self.rn = None
        self.value = value
        self.target_val = None
        self.idx_feat = 0
        self.min_obs = minimum_obs
        self.percentiles = np.arange(10, 100, 20)
        self.max_tree_size = maximum_tree_size
        self.node_height = current_height

    def fit(self, X, y):
        # Check if tree is at maximum height
        if self.max_tree_size is not None and self.max_tree_size < self.node_height + 1:
            self.target_val = np.mean(y)
            self.idx_feat = None
            return self

        n = X.shape[1]
        best_split = [None, None, np.inf]
        for idx in range(n):

            # Skip constant
            if np.nanvar(X[:, idx]) == 0:
                continue

            # Get best split
            for j in self.percentiles:
                perc = np.percentile(a=X[:, idx], q=j)
                sl = X[:, idx] < perc
                sr = X[:, idx] >= perc

                if any([i.sum() == 0 for i in [sl, sr]]):
                    continue

                sl_rsme = rmse(y[sl], np.mean(y[sl]))
                sr_rsme = rmse(y[sr], np.mean(y[sr]))

                if sl_rsme + sr_rsme < best_split[2]:
                    best_split = [idx, perc, sl_rsme + sr_rsme]

        # Split on best
        if best_split[0] is not None:
            self.idx_feat = best_split[0]
            self.value = best_split[1]

            if X[X[:, self.idx_feat] < self.value].shape[0] > self.min_obs:
                self.ln = DT(minimum_obs=self.min_obs,
                             maximum_tree_size=self.max_tree_size,
                             current_height=self.node_height + 1).fit(X[X[:, self.idx_feat] < self.value],
                                                                      y[X[:, self.idx_feat] < self.value])
            else:
                self.target_val = np.mean(y[X[:, self.idx_feat] < self.value])

            if X[X[:, self.idx_feat] >= self.value].shape[0] > self.min_obs:
                self.rn = DT(minimum_obs=self.min_obs,
                             maximum_tree_size=self.max_tree_size,
                             current_height=self.node_height + 1).fit(X[X[:, self.idx_feat] >= self.value],
                                                                      y[X[:, self.idx_feat] >= self.value])
            else:
                self.target_val = np.mean(y[X[:, self.idx_feat] >= self.value])
        else:
            self.target_val = np.mean(y)
            self.idx_feat = None

        return self

    @staticmethod
    def predict(tree, X):
        if tree.ln is None and tree.rn is None:
            return tree.target_val

        if X[tree.idx_feat] < tree.value:
            if tree.ln is not None:
                return tree.predict(tree.ln, X)
            else:
                return tree.target_val
        elif X[tree.idx_feat] >= tree.value:
            if tree.rn is not None:
                return tree.predict(tree.rn, X)
            else:
                return tree.target_val
        else:
            return -1

    @staticmethod
    def print_tree(tree, counter=0, col_names=(), branch='ROOT'):
        # TODO: do BFS for better visualization

        if tree.ln is None and tree.rn is None:
            print(f"FINAL VAL: {tree.target_val}")
        else:
            _id = f"|{branch}|"
            print(_id + "-" * counter + f"{col_names[tree.idx_feat]}({tree.value})")

            if tree.ln is not None:
                tree.print_tree(tree.ln, counter + 1, col_names, 'L')
            else:
                print(f"FINAL VAL: {tree.target_val}")

            if tree.rn is not None:
                tree.print_tree(tree.rn, counter + 1, col_names, 'R')
            else:
                print(f"FINAL VAL: {tree.target_val}")


def _test():
    df = pd.read_csv('../data/cosmetics.csv')

    # Include super feature
    df['super'] = df.Price.apply(np.log)

    train_num = int(np.ceil(df.shape[0]*0.8))
    y = df.Price.values[0:train_num]
    y_test = df.Price.values[train_num::]
    _feats = ['Rank', 'Sensitive', 'super']
    X = df[_feats].values[0:train_num]
    X_test = df[_feats].values[train_num::]

    _minobs = 20
    test = DT(minimum_obs=_minobs, maximum_tree_size=None)
    test = test.fit(X, y)
    test.print_tree(test, col_names=_feats)

    pred_test = make_bulk_prediction(X_test, test)
    pred_train = make_bulk_prediction(X, test)

    print(f"TRAIN RMSE = {rmse(y, pred_train)}")
    print(f"TEST RMSE = {rmse(y_test, pred_test)}")
    print(f"TEST MEAN RMSE = {rmse(y_test, np.mean(y_test))}")
    print(f"TRAIN MEAN RMSE = {rmse(y, np.mean(y))}")




if __name__ == '__main__':
    _test()
