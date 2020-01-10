import numpy as np
import pandas as pd
import sklearn
import sys
import pickle
import csv
import os.path

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier


def main():
    if len(sys.argv) != 3:
        print('Please provide file paths of training, test and output file')
        sys.exit()

    # trained_model = sys.argv[1]
    # features = sys.argv[2]
    testing_filename = sys.argv[1]
    output_filename = sys.argv[2]

    my_path = os.path.abspath(os.path.dirname(__file__))

    trained_model_path = my_path + '/finalized_model.sav'
    features_set_path = my_path + '/features.csv'

    loaded_model = pickle.load(open(trained_model_path, 'rb'))
    df_features = pd.read_csv(features_set_path, index_col=0)
    features_columns = list(df_features.columns)

    df_test = pd.read_csv(testing_filename)

    (ids, test_predictors) = processTestData(df_test)
    print('Test Data has processed :)')
    final_test_predictors = test_predictors[features_columns]

    preprocessed_test_data = dataPreProcessing(final_test_predictors)
    print('Test Data has pre-processed :)')

    predicted = loaded_model.predict(preprocessed_test_data)

    result = pd.DataFrame(ids, columns=['ID'])
    result['Prediction'] = predicted
    result.to_csv(output_filename, index=False)

    print('Predicted output has saved in the output file located at :- ' + output_filename)


def processTestData(testDF):
    ids = testDF['id'].values
    predictors = testDF.drop(['id'], axis=1)
    return ids, predictors


def dataPreProcessing(predictors):
    # Missing data imputation using iterative chained method
    cols = list(predictors.columns.values)
    cleaned_predictors = dataImputation(predictors, cols)

    # Standardized data having 0 mean and unit variance
    standardised_predictors = dataStandardisation(cleaned_predictors)

    # Data Dimensionality reduction to make computation faster and modelling efficient
    transformed_predictors = dataDimensionReduction(standardised_predictors)
    print('The shape of the final predictors set is :-' + str(transformed_predictors.shape))
    return transformed_predictors


def dataImputation(predictors, cols):
    imputer = IterativeImputer(random_state=0, missing_values=np.nan, n_nearest_features=5)
    return pd.DataFrame(imputer.fit_transform(predictors), columns=cols)


def dataStandardisation(predictors):
    scalar = StandardScaler(copy=False)
    return scalar.fit_transform(predictors)


def dataDimensionReduction(predictors):
    nco = predictors.shape[1]
    pca = PCA(n_components=nco)
    return pca.fit_transform(predictors)


if __name__ == '__main__':
    main()
