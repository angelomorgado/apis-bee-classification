# Apis malífera honeybee classification

The main purpose of this project is to classify the presence or absence of honeybees in a given location based on meteriologic data. The data is collected from a CSV file and then processed using a multiple machine learning algorithms. The algorithms used are: K-nearest neighbors, Decision Tree, Random Forest, Support Vector Machine, Logistic Regression and Multilayer Perceptron. The results are then compared and the best algorithm is chosen, using three evaluation metrics: cross-validation, ROC curve and f1-score.

A different way of doing the project, using the torch framework was made as extra work, so it isn't included in the main project and it's not required to run the project.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Project Structure

The main project is located in a Jupyter Notebook file, **main.ipynb** which contains all the code and the results. The data is located in the **dataset_bees.csv** file. The **results** folder contains the results of the algorithms in CSV format.

A seperate way of doing the training and prediction, using the torch framework is located in the **neuralNetwork.py** file. The classes necessary for the neural network are located in the **classes.py** file.

## Dataset

- Latitude and Longitude values shouldn't be used in the model's training. They are only used to plot the data on a map.

- The dataset is split into 2 parts: train and test. The train part is used to train the model, and the test part is used to evaluate the model's performance.

- The first 4 columns are the features, and the last column is the label.

- The features are the following:
  - `mntcm`: the min temperature of the coldest month.
  - `mxtwm`: the max temperature of the hottest month.
  - `rfseas`: the seasonality of rain.
  - `tann`: the average anual temperature.
  - `latitude`: the number of passengers in the taxi ride.
  - `longitude`: the number of passengers in the taxi ride.
  - `y`: the classification label

- The label is the following:
  - `1`: there is presence of bees in the area.
  - `0`: there is pseudo absence of bees in the area.

- The dataset is imbalanced, so it is necessary to use a data balancing algorithm.

- The data is not normalized.
