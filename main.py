import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import rasterio
from rasterio.plot import show
from rasterio.features import rasterize

# Remove all outliers that have a value of 0 and are too close to the ones that have a value of 1
def remove_outliers(data, threshold=20.0):
    # Divide the data into positives and negatives
    data_positives = data[data['y'] == 1]
    data_negatives = data[data['y'] == 0]
    
    # Cycle through all the positives
    for index, row in data_positives.iterrows():
        # Get the latitude and longitude of the positive
        lat = row['latitude']
        lon = row['longitude']

        # Get the euclidian distance between the positive and all the negatives
        distance = np.sqrt((data_negatives['latitude'] - lat)**2 + (data_negatives['longitude'] - lon)**2)
        
        # Remove all the negatives that are too close to the positive
        data_negatives = data_negatives[distance > threshold]
    
    print(data_negatives.shape)

    # Return the new data
    return pd.concat([data_positives, data_negatives])

# Plot the ROC curve
def plot_roc_curve(classifiers, classifiersNames, X_test, y_test):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for classifier, name in zip(classifiers, classifiersNames):
        y_pred = classifier.predict(X_test)
        fpr[name], tpr[name], _ = roc_curve(y_test, y_pred) 
        roc_auc[name] = auc(fpr[name], tpr[name])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    for name in classifiersNames:
        plt.plot(fpr[name], tpr[name], lw=lw, label=name + ' (area = %0.2f)' % roc_auc[name])
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')

# Chose the best classifier using f1 score
def get_best_classifier(classifiers, classifiersNames, X_test, y_test):
    bestClassifier = classifiers[0]
    bestClassifierName = classifiersNames[0]
    bestScore = 0

    for classifier, name in zip(classifiers, classifiersNames):
        y_pred = classifier.predict(X_test)
        score = classifier.score(X_test, y_test)

        if score >= bestScore:
            bestClassifier = classifier
            bestClassifierName = name
            bestScore = score

    # Print the best classifier
    print('Best classifier: ' + bestClassifierName + ' with score: ' + str(bestScore))
    return bestClassifier

# Plot the confusion matrix
def plot_cf(classifier, X_test, y_test):
    # Compute confusion matrix
    # Plot cnf matrix without normalization using ConfusionMatrixDisplay.from_estimator
    # TODO:VER SE É PRECISO
    plt.savefig('ConfusionMatrix.png')
    
#============================================================ MAIN ============================================================
if __name__ == "__main__":
    # Load dataset from csv file
    columnNames = ['mntcm', 'mxtwm', 'rfseas', 'tann', 'latitude', 'longitude', 'y']
    data = pd.read_csv('dataset_bees.csv', names=columnNames)

    # Remove outliers
    data = remove_outliers(data, 23.0)

    # Deal with missing values in dataset (replace with mean value of rows with the same y (label) value)) 
    data = data.fillna(data.groupby('y').transform('mean'))

    # Split dataset into features and labels
    X = data[['mntcm', 'mxtwm', 'rfseas', 'tann']]
    y = data['y']

    # Split the dataset randomly into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize features values to be between 0 and 1 (for better performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a list of classifiers
    classifiersNames = ['KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'Multilayer Perceptron']

    classifiers = [ KNeighborsClassifier(n_neighbors=3),
                    DecisionTreeClassifier(),
                    RandomForestClassifier(n_estimators=100),
                    SVC(),
                    LogisticRegression(),
                    MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
                ]

    # Train each classifier
    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    # Evaluate each classifier using cross validation and f1 score
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for classifier, name in zip(classifiers, classifiersNames):
        scores = cross_val_score(classifier, X, y, cv=skf) # cross_val_score returns the f1 score for each fold of the cross validation (10 folds) 
        print(name + ' : ' + str(scores.mean())) # Print the mean of the f1 scores

    # TODO: Fazer histograma com os scores de cada classificador

    # Plot the ROC curve
    plot_roc_curve(classifiers, classifiersNames, X_test, y_test)

    # Chose the best classifier using f1 score
    bestClassifier = get_best_classifier(classifiers, classifiersNames, X_test, y_test)
    # TODO: Fazer histograma com os scores de cada classificador

    # Predict using the best classifier
    y_pred = bestClassifier.predict(X_test)

    print(classification_report(y_test, y_pred))
