import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# preprocessing function
def preprocessing_data(df: pd.DataFrame):
    data = df.copy()
    # drop NaN values for some columns
    data = data.dropna(subset=['education_level', 'major_discipline', 'experience', 'last_new_job'])
    # Replace other NaN with Unknown value
    data = data.replace(np.nan, 'Unknown')
    # relevent_experience replace with 0 and 1, 1 for having experience and 0 for no experience
    data['relevent_experience'] = data['relevent_experience'].replace(
        ['Has relevent experience', 'No relevent experience'], [1, 0])

    # manually assign ordinal numbers to education_level and company_size
    # for graduate level I will give 1 and for master 2 and for phd 3. Graduate level can be equals to masters and phd but usually people with phd would not represent themselves as graduate.
    # any graduate level certificate can be considered as graduate so I will assign a lower number to graduate than masters.
    # for company_size unknown will get 0.

    data['education_level'] = data['education_level'].replace(['Graduate', 'Masters', 'Phd'], [1, 2, 3])
    data['company_size'] = data['company_size'].replace(
        ['Unknown', '<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'], range(0, 9))

    # convert experience and last_new_job to numeric values
    data['experience'] = data['experience'].str.replace('>', '').str.replace('<', '')
    data['experience'] = pd.to_numeric(data['experience'])

    data['last_new_job'] = data['last_new_job'].str.replace('>', '')
    data['last_new_job'] = data['last_new_job'].replace('never', 0)
    data['last_new_job'] = pd.to_numeric(data['last_new_job'])

    data = pd.get_dummies(data, columns=['company_type', 'enrolled_university', 'gender', 'major_discipline', 'city'])

    # Normalize data using MinMaxScaler function of sci-kit leaern
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_scaled = pd.DataFrame(x_scaled, columns=data.columns)
    return (data_scaled)


def logistic_regression(training_df):
    """
    Setting random_state a fixed value will guarantee that the same sequence of random numbers is generated each
    time you run the code. Split to training and testing datasets.
    Parameters
    ----------
    training_df

    Returns
    -------
    LogisticRegression confusion matrix
    """
    X_train, X_test, y_train, y_test = train_test_split(training_df.drop('target', axis=1),
                                                        training_df['target'], test_size=0.20,
                                                        random_state=42)
    logistic_regression_model = LogisticRegression(C=0.0001, penalty='l2', solver='liblinear', random_state=1,
                                                   max_iter=10000).fit(X_train, y_train)
    predictions = logistic_regression_model.predict(X_test)
    print("Confusion matrix with liblinear LogisticRegression model and l2 penalty. ")
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions)).plot()

    # Use GridSearchCV to tune the parameter of each of the above models.
    # Looks for optimization of hyperparameters over predefined values by fitting the model on the training set
    param_grid = {'C': [0.0001, 0.01, 0.1, 1], 'solver': ['lbfgs', 'liblinear'], 'penalty': ['l1', 'l2'],
                  'random_state': [1, None], 'max_iter': [10000]}
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3)
    grid.fit(X_train, y_train)
    # Parameter setting that gave the best results
    print("The best esimators: ", grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, grid_predictions)).plot()


if __name__ == '__main__':
    raw_data = pd.read_csv("resources/aug_train.csv")
    processed_data = preprocessing_data(raw_data)
    training_df = processed_data.copy()
    logistic_regression(training_df)

    # Drop the column "training_hours"
    new_df = processed_data.drop('training_hours', axis=1)
    logistic_regression(new_df)
