import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
# Classfication
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# Cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import estimate_bandwidth
# Cluster Evaluation
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
# Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import warnings
import os
warnings.filterwarnings(action='ignore')

purpose = 0 # global variable

# -----------------Data Inspection----------------- #
def dataExploration(df):
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    plt.figure(figsize=(10,10))
    sns.countplot(x='Online boarding',hue="satisfaction",data=df,color="green")
    plt.show()
    sns.histplot(x='Age',hue="satisfaction",data=df,kde=True,palette="flare")
    plt.show()
    sns.countplot(x='Customer Type',hue="satisfaction",data=df)
    plt.show()
    sns.histplot(x='Flight Distance',hue="satisfaction",data=df,kde=True,palette="dark")
    plt.show()
    sns.countplot(x='Inflight wifi service',hue="satisfaction",data=df,color="red")
    plt.show()
    sns.countplot(x='Food and drink',hue="satisfaction",data=df,color="orange")
    plt.show()
    plt.figure(figsize = (15,15))
    sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")
    plt.show()

# -----------------Preprocessing---------------- #
def findMissingValue(df):
    # check missing value
    # only 'Arrival Delay in Minutes' has missing values
    #df.dropna(inplace=True)
    df.fillna(df.mean(), inplace = True)
    return df

def encoding(df):
    df.drop(["Unnamed: 0","id"],axis=1,inplace=True)
    df["Gender"] = df["Gender"].map({"Male":1,"Female":0})
    df["Customer Type"] = df["Customer Type"].map({"Loyal Customer":1,"disloyal Customer":0})
    # Type of class
    df.drop(["Type of Travel","Class"],axis=1,inplace=True)
    # Statisfaction
    df["satisfaction"] = df["satisfaction"].map({"satisfied":1,"neutral or dissatisfied":0})
 
    return df

# -----------------Classification---------------- #
def classification(x_train, y_train, x_test, y_test, scalers, models, params_dict):
    best_accuracy = {}
    # find the best parameter by using grid search
    for scaler_key, scaler in scalers.items():
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        print(f'\n[scaler: {scaler_key}]')
        for model_key, model in models.items():
            print(f'\n[model: {model_key}]')

            # grid search
            grid = GridSearchCV(model, param_grid=params_dict[model_key])
            grid.fit(x_train, y_train)
            print(f'parameters: {grid.best_params_}')
            best_model = grid.best_estimator_
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)

            # save the 10 highest accuracy and parameters each models
            list_size = 10
            list_size -= 1
            flag = False

            target_dict = {'accuracy': accuracy, 
                'scaler': scaler_key,
                'model': model_key, 
                'param': grid.best_params_}

            # save accuracy
            if model_key not in best_accuracy.keys():
                best_accuracy[model_key] = []
            if len(best_accuracy[model_key]) <= list_size:
                best_accuracy[model_key].append(target_dict)

            # insert accuracy
            elif best_accuracy[model_key][-1]['accuracy'] < accuracy:
                for i in range(1, list_size):
                    if best_accuracy[model_key][list_size - 1 - i]['accuracy'] > accuracy:
                        best_accuracy[model_key].insert(list_size - i, target_dict)
                        best_accuracy[model_key].pop()
                        flag = True
                        break
                if flag is False:
                    best_accuracy[model_key].insert(0, target_dict)
                    best_accuracy[model_key].pop()

            print(f'accuracy: {accuracy}', end='')
            print()

    return best_accuracy

# -----------------Clustering----------------- #
def featureCombination(df, index):
    if index == 0:
        feature = df[['Gender','Age' ,'satisfaction']]
    elif index == 1:
        feature = df[['Flight Distance', 'Departure/Arrival time convenient', 'satisfaction']]
    elif index == 2:
        feature = df[['Seat comfort', 'Inflight entertainment']]
    return feature

# function for store combination that has the best accuracy
def clustering(df, scalers, models, params_dict):
    best_combination = {}
    best_score = 0
    best_X = 0
    best_label = 0
    
    # Sample Data
    for index in range(3):
        X = featureCombination(df, index)
        feature = X.columns.tolist()
        print(f'\n[feature: {feature}]')

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list() # numerical value

        # find the best parameter by using grid search
        for scaler_key, scaler in scalers.items():
            scaled_X = scaler.fit_transform(X[num_cols])
            print(f'\n[scaler: {scaler_key}]')

            #knee_method(scaled_X)
           
            for model_key, model in models.items():
                print(f'\n[model: {model_key}]')

                cv = [(slice(None), slice(None))]

                # Grid Search
                if (model_key == 'meanshift'): # mean-shift
                    grid = GridSearchCV(estimator=model,
                                        param_grid=estimate_bandwidth(scaled_X),
                                        scoring=silhouette_scorer,
                                        cv=cv)
                else: # other models
                    grid = GridSearchCV(estimator=model,
                                        param_grid=params_dict[model_key],
                                        scoring=silhouette_scorer,
                                        cv=cv)
                    grid.fit(scaled_X)
                
                print(f'best_parameters: {grid.best_params_}')
                score = grid.best_score_
                if (best_score < score):
                    best_score = score
                    best_X = scaled_X
                    best_label = grid.best_estimator_
                    
                    target_dict = {'silhouette': score,
                                    'scaler': scaler_key,
                                    'model': model_key,
                                    'param': grid.best_params_, 
                                    'feature': feature
                                    }
                
                list_size = 10
                list_size -= 1
                flag = False

                # save accuracy
                if model_key not in best_combination.keys():
                    best_combination[model_key] = []
                if len(best_combination[model_key]) <= list_size:
                    best_combination[model_key].append(target_dict)

                # insert accuracy
                elif best_combination[model_key][-1]['silhouette'] < score:
                    for i in range(1, list_size):
                        if best_combination[model_key][list_size - 1 - i]['silhouette'] > score:
                            best_combination[model_key].insert(list_size - i, target_dict)
                            best_combination[model_key].pop()
                            flag = True
                            break
                    if flag is False:
                        best_combination[model_key].insert(0, target_dict)
                        best_combination[model_key].pop()

                print(f'silhouette score: {score}', end='')
                print()

    return best_combination, best_X, best_label

# -----------------Clustering Evaluation----------------- #
def purity_scorer(target, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(target, y_pred)
    score = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels, metric='euclidean')
    return score

def display_silhouette_plot(X, labels):
    sil_score = metrics.silhouette_score(X, labels, metric='euclidean')
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("For n_clusters =", n_clusters_, "The average silhouette score is :", sil_score)
    # compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters_ + 1) * 10])
    y_lower = 10
    for i in range(n_clusters_):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters_)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=sil_score, color="red", linestyle='--')
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for clustering on sample data with n_clusters = %d" % n_clusters_),
                 fontsize=14, fontweight='bold')
    plt.show()
    return sil_score

def knee_method(X):
    nearest_neighbors = NearestNeighbors(n_neighbors=11)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, 10], axis=0)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.savefig("Distance_curve.png", dpi=300)
    plt.title("Distance curve")
    plt.show()
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    fig = plt.figure(figsize=(5, 5))
    knee.plot_knee()
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.show()
    print(distances[knee.knee])

# -------------------------Auto ML------------------------- #
def selectPurpose():
    global purpose
    while(True): # Select Purpose
        purpose = int(input("Select Purpose [1: Classification, 2: Clustering, 3: All] >> "))
        if(0 < purpose and purpose < 4):
            break
        else:
            print("Invalid Value!")
    print(f'purpose: {purpose}')

def selectModel():
    scaler_list = []
    cf_model_list = []
    cl_model_list = []

    while(True): # Select Scalers
        value = int(input("Select Scalers [1: Standard, 2: MinMax, 3: Robust, 4: MaxAbs, 9: All, 0: Exit] >> "))
        if(value == 0): # exit
            if(0 < len(scaler_list)):
                break
            else:
                print("Please choose at least one!")
        if(value == 9): # all
            for i in range (1, 5):
                scaler_list.append(i)
            break
        if(0 < value and value < 5):
            scaler_list.append(value)
        else:
            print("Invalid Value!")
    print('Scaler selected!')

    if(purpose == 1 or purpose == 3):
        while(True): # Select Classification model
            value = int(input("Select Classification model [1: decisionTree_entropy, 2: decisionTree_gini, 3: logistic, 4: svm, 9: All, 0: Exit] >> "))
            if(value == 0): # exit
                if(0 < len(cf_model_list)):
                    break
                else:
                    print("Please choose at least one!")
            if(value == 9): # all
                for i in range (1, 5):
                    cf_model_list.append(i)
                break
            if(0 < value and value < 5):
                cf_model_list.append(value)
            else:
                print("Invalid Value!")
        print('Classification model selected!')

    if(purpose == 2 or purpose == 3):
        while(True): # Select Clustering model
            value = int(input("Select Clustering model [1: kmeans, 2: gmm, 3: dbscan, 4: mean-shift, 9: All, 0: Exit] >> "))
            if(value == 0): # exit
                if(0 < len(cl_model_list)):
                    break
                else:
                    print("Please choose at least one!")
            if(value == 9): # all
                for i in range (1, 5):
                    cl_model_list.append(i)
                break
            if(0 < value and value < 5):
                cl_model_list.append(value)
            else:
                print("Invalid Value!")
        print('Clustering model selected!')

    return scaler_list, cf_model_list, cl_model_list

def setCombination(scaler_list, cf_list = [], cl_list = []):

    # Scaler List
    standard = StandardScaler() #1
    minMax = MinMaxScaler() #2
    robust = RobustScaler() #3
    maxAbs = MaxAbsScaler() #4

    scalers = {}
    cf_models = {}
    cl_models = {}
    cf_params = {}
    cl_params = {}

    for i in scaler_list:
        if (i == 1):
            scalers["standard scaler"] = standard
        elif (i == 2):
            scalers["minMax scaler"] = minMax
        elif (i == 3):
            scalers["robust scaler"] = robust
        elif (i == 4):
            scalers["maxAbs scaler"] = maxAbs

    # Classification Model List
    decisionTree_entropy = tree.DecisionTreeClassifier(criterion="entropy")
    decisionTree_gini = tree.DecisionTreeClassifier(criterion="gini")
    logistic = LogisticRegression()
    svm_model = svm.SVC()

    for i in cf_list:
        if (i == 1):
            cf_models["decisionTree_entropy"] = decisionTree_entropy
            cf_params["decisionTree_entropy"] = {"max_depth": [x for x in range(3, 9, 1)],
                                                "min_samples_split": [x for x in range(2, 10, 1)],
                                                "min_samples_leaf": [x for x in range(3, 10, 1)]}
        elif (i == 2):
            cf_models["decisionTree_gini"] = decisionTree_gini
            cf_params["decisionTree_gini"] = {"max_depth": [x for x in range(3, 9, 1)],
                                            "min_samples_split": [x for x in range(2, 10, 1)],
                                            "min_samples_leaf": [x for x in range(3, 10, 1)]}
        elif (i == 3):
            cf_models["logistic"] = logistic
            cf_params["logistic"] = {"C": [0.001, 0.01],
                                    'penalty': ['l1', 'l2', 'elasticnet', 'none']}
        elif (i == 4):
            cf_models["svm_model"] = svm_model
            cf_params["svm_model"] = {"C": [ 0.1, 1, 10], 
                                    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                                    "gamma": [0.01, 0.1, 1]}

    # Clustering Model List
    kmeans = KMeans() #1
    gmm = GaussianMixture() #2
    dbscan = DBSCAN() #3
    meanshift = MeanShift() #4

    for i in cl_list:
        if (i == 1):
            cl_models["kmeans"] = kmeans
            cl_params["kmeans"] = {"n_clusters": [x for x in range(3, 5)]}
        elif (i == 2):
            cl_models["gmm"] = gmm
            cl_params["gmm"] = {"n_components": [x for x in range(3, 5)]}
        elif (i == 3):
            cl_models["dbscan"] = dbscan
            cl_params["dbscan"] = {"eps": [0.1, 0.5]}
        elif (i == 4):
            cl_models["meanshift"] = meanshift
            cl_params["meanshift"] = {"bandwidth": [1, 2]}

    
    return scalers, cf_models, cf_params, cl_models, cl_params

if __name__ == "__main__":
    # Read data
    dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/'
    train = pd.read_csv(dir+"train.csv")
    test = pd.read_csv(dir+"test.csv")

    # Handle Missing value
    train = findMissingValue(train)
    test = findMissingValue(test)

    # Encoding
    train = encoding(train)
    test = encoding(test)

    # Data exploration
    #dataExploration(train)
    #dataExploration(test)

    # Split Target
    x_train = train.drop(["satisfaction"],axis=1)
    y_train = train['satisfaction']
    x_test = test.drop(["satisfaction"],axis=1)
    y_test = test["satisfaction"]

    # Auto ML
    selectPurpose()
    selected_scaler, selected_cf, selected_cl = selectModel()
    scalers, cf_models, cf_params, cl_models, cl_params = setCombination(selected_scaler, selected_cf, selected_cl)

    # Classification
    if(purpose == 1 or purpose == 3):
        result_dict = classification(x_train, y_train, x_test, y_test, scalers, cf_models, cf_params)
        print("\n-----[Best Classification Result]-----")
        for model_name, result_list in result_dict.items():
            for result in result_list:
                print(result)
            print()

    # Clustering
    if(purpose == 2 or purpose == 3):
        ## get best combination dictionary
        best_result, best_X, best_label = clustering(train, scalers, cl_models, cl_params)
        print("\n-----[Best Clustering Results]-----")
        best_score = 0
        best_combi = 0
        for model_name, result_list in best_result.items():
            print(model_name)
            for result in result_list:
                print(result)
                if (best_score < result['silhouette']):
                    best_score = result['silhouette']
                    best_combi = result
            print()

        print("[Best Combination]")
        print(best_combi)
        #display_silhouette_plot(best_X, best_label.fit_predict(best_X))
    print('\nDone!')
    