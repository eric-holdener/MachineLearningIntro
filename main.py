import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Steps to building a model:
# Define : what type of model are we using? Decision tree? Some other?
# Fit : capture patterns from provided data
# Predict : predict y based off our patterns
# Evaluate : Determine how accurate the model's predictions are

def practice():
    # save filepath to variable
    melbourne_file_path = 'C:\\Users\\ehold\\Desktop\\Folders\\Datasets\\melb_data.csv'

    # read and store data into a dataframe
    melbourne_data = pd.read_csv(melbourne_file_path)

    # drops missing values
    melbourne_data = melbourne_data.dropna(axis=0)

    # print a summary of melbourne data
    print(melbourne_data.describe())

    # print columns in the data
    print(melbourne_data.columns)

    # dot notation for variable we want to predict, price
    y = melbourne_data.Price

    # features are columns used to make predictions - what we are predicting home price off of
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    x = melbourne_data[melbourne_features]

    # double checking our features
    print(x.describe())

    # print first 5 rows of features x, after dropping empty values
    print(x.head())

    # define the model and specify random state so results are always the same
    melbourne_model = DecisionTreeRegressor(random_state=1)

    # fit model using features and variable to predict
    melbourne_model.fit(x, y)

    print('Making predictions for the following 5 houses:')
    print(x.head())
    print('The predictions are')
    # Takes model we are using (decision tree), and what features to use (in this case the head of the features x)
    print(melbourne_model.predict(x.head()))

    # MAE (mean absolute error) : The absolute value of the prediction error, averaged across every prediction
    predicted_home_prices = melbourne_model.predict(x)
    print(mean_absolute_error(y, predicted_home_prices))

    # in sample scope - single sample to build the model and evaluate it
    # need to measure performance on data that the model hasnt seen before - validation data
    # train test split to get training data for x and y, and validation data for x and y
    train_X, val_X, train_y, val_y = train_test_split(x, y, random_state= 0)

    # define model
    melbourne_model = DecisionTreeRegressor()

    # fit the model using the training data for x and y
    melbourne_model.fit(train_X, train_y)

    # make predictions using the validation data for x
    val_predictions = melbourne_model.predict(val_X)

    # compare predictions using x validation to the actual y validation values
    print(mean_absolute_error(val_y, val_predictions))

    # overfitting - when you run too many splits on the training data, leaving too little individual data points in the end nodes to train off of, resulting in predictions that are too close to the training sets values
    # underfitting - when the groups still have too many individual data points it cannot capture the important distinctions and patterns in the data (not even splits in the decision tree)

def initialize_data():
    melbourne_file_path = 'C:\\Users\\ehold\\Desktop\\Folders\\Datasets\\melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    x = filtered_melbourne_data[melbourne_features]
    train_X, val_X, train_y, val_y = train_test_split(x, y, random_state= 0)
    return(train_X, val_X, train_y, val_y)

def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def main():
    train_X, val_X, train_y, val_y = initialize_data()
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))


main()
