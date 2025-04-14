import numpy as np
import pandas as pd


class Node():
    """*Class* untuk menyimpan informasi tentang *node* dalam *decision tree*.
    
    **Constructor**:
        `__init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None)`
    """
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None):
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Leaf node
        self.value = value

class DecisionTreeClassifier():
    """*Class* untuk melakukan klasifikasi dengan model *decision tree*.
    
    **Constructor**:
        `__init__(self, min_samples_split = 2, max_depth = 10)`
    
    **Method**:
        `fit(self, X, y)`: Melakukan fitting model
        `predict(self, X)`: Memprediksi sebuah dataset
        `make_prediction(self, x, tree)`: Melakukan prediksi hanya untuk sebuah data
        `print_tree(self, node=None, depth=0)`: Mencetak decision tree
    """
    def __init__(self, min_samples_split = 2, max_depth = 10):
        self.root = None
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth = 0):
        """Membuat decision tree secara rekursif.

        Args:
            dataset (_type_): Dataset yang akan digunakan untuk membangun tree.
            curr_depth (int, optional): Kedalaman saat ini dari tree. Defaults to 0.
        """
        
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        
        # Nilai decision node
        if num_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
        
        # Nilai leaf node
        leaf_value = self.calculate_leaf_value(y)
        return Node(value = leaf_value)
    
    def get_best_split(self, dataset, num_features):
        """Mencari split terbaik dari suatu dataset.

        Args:
            dataset (_type_): Dataset untuk mencari split terbaik.
            num_features (_type_): Banyak fitur dalam dataset.
        """
        
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                left_subtree, right_subtree = self.split(dataset, feature_index, threshold)
                
                if len(left_subtree) > 0 and len(right_subtree) > 0:
                    y = dataset[:, -1]
                    curr_info_gain = self.information_gain(y, left_subtree[:, -1], right_subtree[:, -1], mode='gini')
                    
                    if curr_info_gain > max_info_gain:
                        max_info_gain = curr_info_gain
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left"] = left_subtree
                        best_split["right"] = right_subtree
                        best_split["info_gain"] = max_info_gain
        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        """Membagi data menjadi dua subset kiri dan kanan berdasarkan threshold.

        Args:
            dataset (_type_): Dataset untuk membagi data.
            feature_index (_type_): Index fitur yang akan digunakan untuk membagi data.
            threshold (_type_): Nilai threshold.
        """
        
        left_subtree = np.array([row for row in dataset if row[feature_index] <= threshold])
        right_subtree = np.array([row for row in dataset if row[feature_index] > threshold])
        
        return left_subtree, right_subtree
    
    def information_gain(self, parent, left_child, right_child, mode='entropy'):
        """Menghitung information gain dari suatu split menggunakan entropi atau gini index.
        Args:
            parent (_type_): Dataset sebelum split.
            left_child (_type_): Dataset setelah split ke kiri.
            right_child (_type_): Dataset setelah split ke kanan.
            mode (str, optional): Mode perhitungan (`entropy` atau `gini`). Defaults to `entropy`.
        """
                
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        if mode == 'gini':
            gain = self.gini(parent, left_child, right_child, weight_left, weight_right)
        else:
            gain = self.entropy(parent, left_child, right_child, weight_left, weight_right)
        
        return gain
    
    def entropy(self, y):
        """Menghitung entropi label.

        Args:
            y (_type_): Label dari dataset.
        """
        
        class_labels = np.unique(y)
        entropy = 0
        
        for label in class_labels:
            prob = len(y[y == label]) / len(y)
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def gini_index(self, y):
        """Menghitung gini index label.

        Args:
            y (_type_): Label dari dataset.
        """
        
        class_labels = np.unique(y)
        gini = 1
        
        for label in class_labels:
            prob = len(y[y == label]) / len(y)
            gini -= prob ** 2
        
        return gini
    
    def calculate_leaf_value(self, y):
        """Menghitung nilai leaf node.

        Args:
            y (_type_): Label dari dataset.
        """
        
        Y = list(y)
        return max(Y, key=Y.count)
    
    def print_tree(self, node=None, depth=0):
        """Mencetak decision tree.

        Args:
            node (_type_, optional): Node yang akan dicetak. Defaults to None.
            depth (int, optional): Kedalaman saat ini dari tree. Defaults to 0.
        """
        
        if node is None:
            node = self.root
        
        if node.value is not None:
            print(f"{' ' * depth * 2}Leaf Node: {node.value}")
            return
        
        print(f"{' ' * depth * 2}Node: [Feature Index: {node.feature_index}, Threshold: {node.threshold}]")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)
        
    def fit(self, X, y):
        """Melakukan fitting model dengan dataset yang diberikan.

        Args:
            X (_type_): _Dataframe_ fitur dari dataset.
            y (_type_): _Label_ dari dataset.
        """
        
        dataset = np.column_stack((X, y))
        self.root = self.build_tree(dataset)
        
    def predict(self, X):
        """Memprediksi sebuah dataset berdasarkan model yang telah dilatih.

        Args:
            X (_type_): _Dataframe_ fitur dari dataset yang akan diprediksi.
        
        Returns:
            _type_: _Array_ dari hasil prediksi.
        """
        
        predictions = [self._predict(x, self.root) for x in X]
        return np.array(predictions)
    
    def make_prediction(self, x, tree):
        """Melakukan prediksi hanya untuk sebuah data.

        Args:
            X (_type_): _description_
            tree (_type_): _description_
            
        Returns:
            _type_: Hasil prediksi untuk sebuah data.
        """
        
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)