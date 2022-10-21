import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Remove all outliers that have a value of 0 and are too close to the ones that have a value of 1
def remove_outliers(data, threshold=0.5):
    data_positives = data[data['y'] == 1]
    data_negatives = data[data['y'] == 0]

    # Remove all negatives that are close to the positives by using the threshold in a 2d space of latitude and longitude
    
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

if __name__ == "__main__":
    # Load dataset from csv file
    columnNames = ['mntcm', 'mxtwm', 'rfseas', 'tann', 'latitude', 'longitude', 'y']
    data = pd.read_csv('dataset_bees.csv', names=columnNames)

    # Remove outliers
    data = remove_outliers(data, 20.0)

    # Split dataset into features and labels
    X = data[['mntcm', 'mxtwm', 'rfseas', 'tann']]
    y = data['y']

    # Deal with missing values in dataset (replace with mean)
    X = X.fillna(X.mean())
    
    # Normalize features values to be between 0 and 1 (for better performance)
    #X = (X - X.mean()) / X.std()

    # Split dataset into training and testing using cross validation    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a list of classifiers
    classifiersNames = ['KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']

    classifiers = [ KNeighborsClassifier(n_neighbors=3),
                    DecisionTreeClassifier(),
                    RandomForestClassifier(n_estimators=100),
                    SVC(),
                    LogisticRegression() 
                ]

    # Train each classifier
    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    # Predict using each classifier
    for classifier, name in zip(classifiers, classifiersNames):
        y_pred = classifier.predict(X_test)
        #print(y_pred)
        #print(y_test)
        print(name + ' : '+ str(classifier.score(X_test, y_test)))