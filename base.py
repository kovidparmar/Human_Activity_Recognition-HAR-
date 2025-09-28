import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import entropy, information_gain, gini_index, gini_gain, variance_gain, loss

np.random.seed(42)

class Node:
    def __init__(self):
        # pred attribute for the node
        self.feature_label = None
        # pred value of pred attribute for the node
        self.pred_value = None
        # all the children nodes
        self.children = dict()
        # for real input cases left and right nodes
        self.left = None
        self.right = None
        # partition value storage
        self.split_value = None

class DecisionTree():
    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

        # to specify which case to consider for prediction and plotting
        self.in_type = None
        self.out_type = None

    def fit(self, X, y):
        category = "category"

        lst = []
        for i in list(X.columns):
            lst.append(X[i].dtype.name)

        self.in_type = lst[0]
        self.out_type = y.dtype.name

        # case: DIDO
        if ((category in lst) and y.dtype.name == "category"):
            features = list(X.columns)
            out = "out"
            if (out in features):
                features.remove(out)
            temp_X = X
            temp_X['out'] = y
            feature_mat = temp_X
            depth = 0
            self.tree = self.DIDO(feature_mat, None, features, 0)

        # case: DIRO
        elif ((category in lst) and y.dtype.name != "category"):
            features = list(X.columns)
            out = "out"
            if (out in features):
                features.remove(out)
            temp_X = X
            temp_X['out'] = y
            feature_mat = temp_X
            depth = 0
            self.tree = self.DIRO(feature_mat, None, features, 0)

        # case: RIDO
        elif ((category not in lst) and y.dtype.name == "category"):
            out = 'out'
            if (out in X.columns):
                X = X.drop(['out'], axis=1)
            self.X = X
            self.y = y
            self.no_of_attributes = len(list(X.columns))
            self.no_of_out_classes = len(set(y))
            feature_mat = X
            output_vec = y
            depth = 0
            self.tree = self.RIDO(feature_mat,output_vec, depth)

        # case: RIRO
        elif ((category not in lst) and y.dtype.name != "category"):
            out = 'out'
            if (out in X.columns):
                X = X.drop(['out'], axis=1)
            self.no_of_attributes = len(list(X.columns))
            feature_mat = X
            output_vec = y
            depth = 0
            self.tree = self.RIRO(feature_mat,output_vec, depth)

    def _split_riro(self, X, y):
        m = list(np.unique(y))
        if(len(m) <= 1):
            return m[0], None

        start_loss = 10**8
        best_feature, best_split_threshold = None, None

        for feature in list(X.columns):

            a = X[feature]
            a = pd.DataFrame(a)
            a['out'] = y
            a = a.sort_values(by=feature, ascending=True)
            a = a.reset_index()
            a = a.drop(['index'], axis=1)

            classes = a['out']
            a = a.drop(['out'],axis=1)
            cutoff_values = a

            for i in range(1, len(classes)):
                c = classes[i-1]

                curr_loss = loss(classes, i-1)

                if (curr_loss < start_loss):
                    start_loss = curr_loss
                    best_feature = feature
                    best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
        return best_feature, best_split_threshold

    def RIRO(self, samples, output_vec, depth=0, parent_node=None):

        if depth < self.max_depth:
            feature, split_value = self._split_riro(samples, output_vec)

            if (feature is not None and split_value is not None):
                samples['out'] = output_vec
                samples = samples.sort_values(by=feature, ascending=True)

                samples = samples.reset_index()

                samples = samples.drop(['index'], axis=1)

                output_vec = samples['out']
                samples = samples.drop(['out'],axis=1)

                X_l = list()
                y_l = list()
                X_r = list()
                y_r = list()

                for index in range(len(samples)):
                    if (samples.loc[index, feature] <= split_value):
                        X_l.append(samples.loc[index])
                        y_l.append(output_vec[index])
                    else:
                        X_r.append(samples.loc[index])
                        y_r.append(output_vec[index])

                X_l = pd.DataFrame(X_l)
                X_r = pd.DataFrame(X_r)
                y_l = pd.Series(y_l)
                y_r = pd.Series(y_r)

                X_l = X_l.reset_index()
                X_r = X_r.reset_index()
                y_l = y_l.reset_index()
                y_r = y_r.reset_index()

                X_r = X_r.drop(['index'], axis=1)
                X_l = X_l.drop(['index'], axis=1)
                y_r = y_r.drop(['index'], axis=1)
                y_l = y_l.drop(['index'], axis=1)

                node = Node()
                node.feature_label = feature
                node.split_value = split_value
                node.pred_value = round(float(output_vec.mean()),6)

                if (len(X_l)!=0 and len(y_l)!=0):
                    node.left = self.RIRO(X_l, y_l, depth+1, node)
                if (len(X_r)!=0 and len(y_r)!=0):
                    node.right = self.RIRO(X_r, y_r, depth+1, node)

            elif (feature is not None and split_value is None):
                node = Node()
                node.pred_value = feature
                return node
            return node
        
        else:
            node = Node()
            node.pred_value = round(float(output_vec.mean()),6)
            return node

    def _best_split(self, X, y):

        m = list(np.unique(y))
        if(len(m) == 1):
            return m[0], None

        best_feature, best_split_threshold = None, None

        if (self.criterion == "information_gain"):
            start_gain = -10**8
        else:
            start_gain = 10**8

        for feature in list(X.columns):

            a = X[feature]
            a = pd.DataFrame(a)
            a['out'] = y
            a = a.sort_values(by=feature, ascending=True)

            a = a.reset_index()

            a = a.drop(['index'], axis=1)

            classes = a['out']

            temp_c = np.unique(classes, return_counts=True)
            temp_classes = list(temp_c[0])
            classes_count = list(temp_c[1])

            self.no_of_out_classes = len(temp_classes)

            a = a.drop(['out'],axis=1)
            cutoff_values = a

            labels_before = dict()
            for i in range(self.no_of_out_classes):
                labels_before[temp_classes[i]] = 0

            labels_after = dict()

            for elem in range(self.no_of_out_classes):
                labels_after[temp_classes[elem]] = classes_count[elem]  

            for i in range(1, len(classes)):
                c = classes[i-1]
                labels_before[c]+=1
                labels_after[c]-=1

                if (self.criterion == "information_gain"):
                    gain_left = entropy(pd.Series(list(labels_before.values())))
                    gain_right = entropy(pd.Series(list(labels_after.values())))
                    gain_temp = entropy(y) - (i * gain_left + (len(classes) - i) * gain_right) / len(classes)
                    
                    if (start_gain < gain_temp):
                        start_gain = gain_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
                else:
                    gini_left = gini_index(pd.Series(list(labels_before.values())))
                    gini_right = gini_index(pd.Series(list(labels_after.values())))
                    gini_index_temp = (i * gini_left + (len(classes) - i) * gini_right) / len(classes)

                    if (gini_index_temp <= start_gain):
                        start_gain = gini_index_temp
                        best_feature = feature
                        best_split_threshold = round(((cutoff_values.loc[i-1,feature] + cutoff_values.loc[i,feature])/2), 6)
        return best_feature, best_split_threshold

    def RIDO(self, samples, output_vec, depth=0, parent_node=None):

        if depth < self.max_depth:
            feature, split_value = self._best_split(samples, output_vec)
            if (feature is not None and split_value is not None):

                samples['out'] = output_vec
                samples = samples.sort_values(by=feature, ascending=True)

                samples = samples.reset_index()

                samples = samples.drop(['index'], axis=1)

                output_vec = samples['out']
                samples = samples.drop(['out'],axis=1)

                X_l = list()
                y_l = list()
                X_r = list()
                y_r = list()

                for index in range(len(samples)):
                    if (samples.loc[index, feature] <= split_value):
                        X_l.append(samples.loc[index])
                        y_l.append(output_vec[index])
                    else:
                        X_r.append(samples.loc[index])
                        y_r.append(output_vec[index])

                X_l = pd.DataFrame(X_l)
                X_r = pd.DataFrame(X_r)
                y_l = pd.Series(y_l)
                y_r = pd.Series(y_r)

                X_l = X_l.reset_index()
                X_r = X_r.reset_index()
                y_l = y_l.reset_index()
                y_r = y_r.reset_index()

                X_r = X_r.drop(['index'], axis=1)
                X_l = X_l.drop(['index'], axis=1)
                y_r = y_r.drop(['index'], axis=1)
                y_l = y_l.drop(['index'], axis=1)

                node = Node()
                node.feature_label = feature
                node.split_value = split_value
                node.pred_value = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                
                if (len(X_l)!=0 and len(y_l)!=0):
                    node.left = self.RIDO(X_l, y_l, depth+1, node)
                if (len(X_r)!=0 and len(y_r)!=0):
                    node.right = self.RIDO(X_r, y_r, depth+1, node)
            
            elif (feature is not None and split_value is None):
                node = Node()
                node.pred_value = feature
                return node
            return node
        
        else:
            if (len(output_vec) == 0):
                node = Node()
                node.pred_value = None
                return node
            temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
            node = Node()
            node.pred_value = temp
            return node

    def DIRO(self, samples, target_attr, attributes, depth):

        output_vec = samples['out']

        if (depth < self.max_depth):
            if (len(output_vec.unique()) <= 1):
                temp = output_vec.unique()[0]
                return temp
            
            elif (len(attributes) == 0):
                temp = sum(output_vec)/len(output_vec)
                return temp
            
            else:
                a = list()

                for x in attributes:
                    attr = samples[x]
                    var_gain = variance_gain(output_vec, attr)
                    a.append(var_gain)

                best_attr = attributes[a.index(max(a))]

                root = Node()
                root.feature_label = best_attr

                for x in samples[best_attr].unique():
                    new_data = samples[samples[best_attr]==x]
                    new_data = new_data.reset_index()
                    new_data = new_data.drop(['index'], axis=1)

                    if (len(new_data) == 0):
                        root.children[x] = sum(output_vec)/len(output_vec)
                    else:
                        temp_attr = []
                        for y in attributes:
                            if (y!=best_attr):
                                temp_attr.append(y)

                        subtree = self.DIRO(new_data, best_attr, temp_attr, depth+1)

                        root.children[x] = subtree

                return root
        else:
            temp = sum(output_vec)/len(output_vec)
            return temp

    def DIDO(self, samples, target_attr, attributes, depth):
        """For Discrete input, discrete output
            Output: a tree
        """
        output_vec = samples['out']

        if (depth < self.max_depth):

            if (len(list(output_vec.unique())) <= 1):
                temp = list(output_vec.unique())[0]
                return temp

            elif (len(attributes) == 0):
                temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                return temp

            else:
                a = list()
                if (self.criterion == "information_gain"):
                    for x in attributes:
                        attr = samples[x]

                        inf_gain = information_gain(output_vec,attr)

                        a.append(inf_gain)
                    best_attr = attributes[a.index(max(a))]

                elif (self.criterion == "gini_index"):
                    for x in attributes:
                        attr = samples[x]

                        g_gain = gini_gain(output_vec,attr)

                        a.append(g_gain)
                    best_attr = attributes[a.index(max(a))]

                root = Node()
                root.feature_label = best_attr
                
                for x in samples[best_attr].unique():
                    new_data = samples[samples[best_attr]==x]

                    if (len(new_data) == 0):
                        root.children[x] = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
                    else:
                        temp_attr = []
                        for y in attributes:
                            if (y!=best_attr):
                                temp_attr.append(y)

                        subtree = self.DIDO(new_data, best_attr, temp_attr, depth+1)

                        root.children[x] = subtree

                return root
        else:
            temp = np.unique(output_vec)[np.argmax(np.unique(output_vec,return_counts=True)[1])]
            return temp

    def predict(self, X):

        # case: DIDO
        if (self.in_type == "category" and self.out_type == "category"):
            y_hat = list()

            if (type(self.tree) != Node):
                y_hat.append(self.tree)
                return pd.Series(y_hat)

            attributes = list(X.columns)
            attributes.remove('out')

            for i in range(len(X)):
                tree = self.tree
                data = list(X.loc[i])
                while(1):
                    curr_feat = tree.feature_label
                    curr_val = data[curr_feat]
                    if (type(tree.children[curr_val]) == Node):
                        tree = tree.children[curr_val]
                    else:
                        y_hat.append(tree.children[curr_val])
                        break

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: DIRO
        elif (self.in_type == "category" and self.out_type != "category"):
            y_hat = list()

            attributes = list(X.columns)
            attributes.remove('out')

            for i in range(len(X)):
                tree = self.tree
                data = list(X.loc[i])
                while(1):
                    curr_feat = tree.feature_label
                    curr_val = data[curr_feat]
                    if (type(tree.children[curr_val]) == Node):
                        tree = tree.children[curr_val]
                    else:
                        y_hat.append(tree.children[curr_val])
                        break

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: RIDO
        elif (self.in_type != "category" and self.out_type == "category"):
            y_hat = list()

            attributes = list(X.columns)
            out = 'out'
            if (out in attributes):
                attributes.remove('out')

            for i in range(len(X)):
                tree = self.tree
                while(1):
                    curr_node_feature = tree.feature_label
                    if (curr_node_feature==None):
                        break
                    sample_val = X.iloc[i, curr_node_feature]
                    if (sample_val <= tree.split_value):
                        if (tree.left != None):
                            tree = tree.left
                        else:
                            break
                    else:
                        if (tree.right != None):
                            tree = tree.right
                        else:
                            break

                y_hat.append(tree.pred_value)

            y_hat = pd.Series(y_hat)
            return y_hat

        # case: RIRO
        elif (self.in_type != "category" and self.out_type != "category"):
            y_hat = list()

            for i in range(len(X)):
                tree = self.tree
                while(1):
                    curr_node_feature = tree.feature_label
                    if (curr_node_feature==None):
                        break
                    sample_val = X.loc[i, curr_node_feature]
                    if (sample_val <= tree.split_value):
                        if (tree.left != None):
                            tree = tree.left
                        else:
                            break
                    else:
                        if (tree.right != None):
                            tree = tree.right
                        else:
                            break
                
                y_hat.append(tree.pred_value)

            y_hat = pd.Series(y_hat)
            return y_hat

    def plot(self):

        if self.in_type == "category":

            tree = self.tree

            def printdict(d, indent=0):

                print("\t"*(indent-1) + "feature:" +str(d.feature_label))

                for key, value in d.children.items():

                    print('\t' * indent + "\t" + "feat_value:" +  str(key))

                    if isinstance(value, Node):
                        printdict(value, indent+1)
                    else:
                        print('\t' * (indent+1) + str(value))

            printdict(tree)

        elif (self.in_type != "category"):
            tree = self.tree
            def printdict(d, indent=0):

                if (isinstance(d.left, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.left, indent+1)

                if (isinstance(d.right, Node)):
                    print("\t"*(indent) + "feature:" +str(d.feature_label) + "\t" + "split value:" + str(d.split_value))
                    printdict(d.right, indent+1)

                if (isinstance(d.right, Node) == False and isinstance(d.left, Node) == False):
                    print('\t' * (indent+1) + str(d.pred_value))

            printdict(tree)