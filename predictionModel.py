import pandas as pd
import numpy as np
import sklearn.preprocessing as sklp
import sklearn.model_selection as skms
import sklearn.tree as sktree


def loadPreprocessData(weatherdatacsv):

    # Importing Beach Dataset

    weatherdata = pd.read_csv(weatherdatacsv, sep=',')
    weatherdata.columns = ['desc', 'daytime', 'temperature', 'pressure',
                           'humidity', 'wind_str', 'wind_deg', 'beachday?']

    # Dropping Lines With Missing Values

    weatherdata = weatherdata.dropna()

    # Encoding Categorical Values

    weatherEncoded = weatherdata.copy()
    encoder = sklp.LabelEncoder()
    weatherEncoded['desc'] = (encoder.fit_transform(weatherEncoded['desc']))
    weatherEncoded['beachday?'] = weatherEncoded['beachday?'].astype(int)

    # Let's return the encoder aswell so we can use it to encode the weather we want to predict for today

    return weatherEncoded, encoder

def modelTraining(weatherEncoded):

    # Splitting Dataset Into Training and Testing Sets

    X = weatherEncoded.drop(['beachday?'], axis=1)
    Y = weatherEncoded['beachday?']
    SEED = 7
    X_train, X_test, Y_train, Y_test = skms.train_test_split(X, Y, test_size=0.3, random_state= SEED, stratify= Y)

    # Testing the optimal parameters for the model tree leaf size

    treeLeaves = np.arange(2,40)
    tree_leaves_train_accuracy =np.empty(len(treeLeaves))
    tree_leaves_test_accuracy = np.empty(len(treeLeaves))

    for i,k in enumerate(treeLeaves):
        DecisionModel = sktree.DecisionTreeClassifier(criterion = "gini", max_leaf_nodes= k, random_state= 0, max_depth= 5)
        DecisionModel.fit(X_train, Y_train)
        tree_leaves_train_accuracy[i] = DecisionModel.score(X_train, Y_train)
        tree_leaves_test_accuracy[i] = DecisionModel.score(X_test, Y_test)
    
    # Choosing the optimal leaf size

    treeHighestAccurracyIndex = (np.argmax(tree_leaves_test_accuracy, axis= 0))
    treeHighestAccurracyLeafSize = treeLeaves[treeHighestAccurracyIndex]

    # Training the model with the optimal parameters

    DecisionModel = sktree.DecisionTreeClassifier(criterion = "gini", max_leaf_nodes= treeHighestAccurracyLeafSize, random_state= 0, max_depth= 5)
    DecisionModel.fit(X_train, Y_train)

    # Returning the model to predict today's weather

    return DecisionModel