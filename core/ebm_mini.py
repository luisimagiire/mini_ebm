# TODO: Implement EBM Mini
from decision_tree_mini import DT

class EBM:
    def __init__(self, num_iter=10, max_tree_size=3, learning_rate=0.1):
        self.M = num_iter
        self.max_node = max_tree_size
        self.lr = learning_rate

    def fit(self, X, y):
        models_map = dict()
        feats_num = X.shape[1]

        for m in range(self.M):
            for feat in range(feats_num):
                model = DT()


        pass

    def predict(self, X):
        pass
