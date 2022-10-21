# How the dataset is structured

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

- The dataset is imbalanced, so the model's performance is evaluated using the F1 score.

- The data is not normalized.
