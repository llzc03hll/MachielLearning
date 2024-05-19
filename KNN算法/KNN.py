import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#KNN算法实现
class KNearestNeighbor(object):
    def __init__(self) :
        pass

    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X,k=1):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        d1 = -2 * np.dot(X, self.X_train.T)   
        d2 = np.sum(np.square(X), axis=1, keepdims=True)   
        d3 = np.sum(np.square(self.X_train), axis=1)   
        dist = np.sqrt(d1 + d2 + d3)

        # 根据K值，选择最可能属于的类别
        y_pred = np.zeros(num_test)
        for i in range(num_test):
           dist_k_min = np.argsort(dist[i])[:k]    # 最近邻k个实例位置
           y_kclose = self.y_train[dist_k_min]     # 最近邻k个实例对应的标签
           y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))    # 找出k个标签中从属类别最多的作为预测类别

        return y_pred

# 加载数据集
def load_datasets():
    iris = datasets.load_iris()
    wine = datasets.load_wine()
    zoo_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
    zoo_data = pd.read_csv(zoo_url, header=None)
    zoo = {
        'data': zoo_data.iloc[:, 1:-1].values,
        'target': zoo_data.iloc[:, -1].values - 1,  # 将标签从1开始的类别转换为从0开始
        'target_names': np.unique(zoo_data.iloc[:, -1])
    }

    glass_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    column_names = [
    "Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"
    ]
    glass_data = pd.read_csv(glass_url,names=column_names)
    glass = {
        'data' : glass_data.iloc[:,1:-1].values,
        'target':glass_data.iloc[:, -1].values - 1,  # 将标签从1开始的类别转换为从0开始
        'target_names': np.unique(glass_data.iloc[:, -1])
    }
    return iris, wine, zoo,glass

# 预处理数据
def preprocess_data(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data['data'])
    y = data['target']
    return X, y

# 交叉验证函数
def cross_validation(model, X, y, k=3, n_splits=10, n_repeats=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    total_iterations = n_splits * n_repeats
    with tqdm(total=total_iterations, desc="Cross-validation progress") as pbar:
        for _ in range(n_repeats):
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.train(X_train, y_train)
                y_pred = model.predict(X_test, k=k)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
                
                pbar.update(1)

    return np.mean(accuracies), np.std(accuracies)

# 主函数
def main():
    iris, wine, zoo, glass = load_datasets()

    datasets_list = [iris, wine, zoo,glass]
    dataset_names = ["Iris", "Wine", "Zoo","Glasss"]

    for data, name in zip(datasets_list, dataset_names):
        X, y = preprocess_data(data)

        knn = KNearestNeighbor()
        mean_accuracy, std_accuracy = cross_validation(knn, X, y, k=3, n_splits=10, n_repeats=10)

        print(f"Dataset: {name}")
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Standard Deviation: {std_accuracy:.2f}")

if __name__ == "__main__":
    main()