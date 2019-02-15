import matplotlib.pyplot as plt
import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

# ------------------------------------------------------------------------------
DOWLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ------------------------------------------------------------------------------
housing = load_housing_data()
housing.head()

housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

housing.hist(bins=50, figsize=(20, 15))
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

housing["income_cat"].hist(bins=50, figsize=(10, 5))

housing["income_cat"].value_counts() / len(housing)

test_set["income_cat"].value_counts() / len(test_set)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# dropping the income category so that the dataset goes back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# making a copy of the training set to explore it without changing the original
housing = strat_train_set.copy()

# scatterplotting data using longitude and latitude
housing.plot(kind="scatter", x="longitude", y="latitude")

# Now with alpha set to 0.1
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

# Looking for correlations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# Combining attributes to explore more correlations
# A custom transformer CombinedAttributesAdder will handle this
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# We observe that bedrooms_per_room is more correlated than total_rooms and total_bedrooms

# ----- Preparing the data for the ML algorithms ------
housing = strat_train_set.drop("median_house_value", axis=1)
# drop() returns a copy and does not alter strat_train_set
housing_labels = strat_train_set["median_house_value"].copy()
# What we've done here is split the predictors from the labels

# Cleaning the data by replacing missing values with the median using the Imputer from Scikit-Learn
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
# fit() computes the median of each attribute and returns the result in a variable called statistics_ that belongs to Imputer
X = imputer.transform(housing_num)
# transform() returns a numpy array containing all values and median ones where values where missing
# From Numpy array back to Pandas dataframe:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Converting text categories to numerical representation
housing_cat = housing["ocean_proximity"]
encoder = CustomLabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)

# -----------------------------------------------------------------------------
# Custom transformer class
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# -----------------------------------------------------------------------------
# There's a problem between the book implementation of LabelBinarizer and the current version of it
# Working around this by using a custom LabelBinarizer


class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output

    def fit(self, X, y=None):
        self.enc = LabelBinarizer(sparse_output=self.sparse_output)
        self.enc.fit(X)
        return self

    def transform(self, X, y=None):
        return self.enc.transform(X)

# -----------------------------------------------------------------------------
# To enable sending dataframe as argument to transformers instead of numpy array
# Let's make a Class to handle the conversion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# -----------------------------------------------------------------------------
# Pipelines
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)), ('imputer', Imputer(
    strategy="median")), ('attribs_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler())])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('custom_label_binarizer', CustomLabelBinarizer()), ])

# A pipeline of pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
# -----------------------------------------------------------------------------
# Preparing the data by using the full_pipeline
housing_prepared = full_pipeline.fit_transform(housing)
# -----------------------------------------------------------------------------

# Training
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


housing_predictions = lin_reg.predict(housing_prepared)
# Measuring accuracy using the rmse (root mean squered error)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# Decision tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Cross validation to train and evaluate on a validation set
# Scikit provides k-fold cross validation, splitting in ten the training set
# Each time training on 9 sets and evaluating on 1 (validation set)

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                              scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)


def display_scores(tree_scores):
    print("Scores:", tree_scores)
    print("Mean:", tree_scores.mean())
    print("Standard deviation:", tree_scores.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Random Forest

# Train using fit(prepared_data, labels)
forest_reg = RandomForestRegressor()
# Train using fit(prepared_data, labels)
forest_reg.fit(housing_prepared, housing_labels)
# Evaluate on the same training data set
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# Evaluate using cross validation
forest_scores = cross_val_score(forest_reg, housing_prepared,
                                housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
