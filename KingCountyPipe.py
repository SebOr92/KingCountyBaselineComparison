import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def load_and_check_data(path):

    data = pd.read_csv(path,
                       sep=',')
    print(data.shape)
    print(data.head())
    print(data.dtypes)
    print(data.columns)
    print(data.isnull().sum())
    print(data.describe())
    print("Successfully loaded data from CSV")
    return data

def preprocess_data(data, seed):

    X = data.drop(['id', 'date', 'price'], axis = 1)
    y = data['price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

def baseline_comparison(data, pipes, cv):
    
    rmse_mean, r2_mean, names = [], [], []

    for name, model in pipes:
        rmse_res = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'neg_mean_squared_error')
        print("{name} RMSE finished".format(name = name))
        rmse_ouput = "%s: %f (+/- %f)" % (name, rmse_res.mean(),  rmse_res.std())
        print(rmse_ouput)
        rmse_mean.append(rmse_res.mean())

        r2_res =cross_val_score(model, X_train, y_train, cv = cv, scoring = 'r2')
        print("{name} Rsquared finished".format(name = name))
        r2_output = "%s: %f (+/- %f)" % (name, r2_res.mean(),  r2_res.std())
        print(r2_output)
        r2_mean.append(r2_res.mean())
        names.append(name)

    cv_final_train = pd.DataFrame(
        {'RMSEMeans': rmse_mean,
         'RSquaredMeans': r2_mean,
         'Model': names}).sort_values('RSquaredMeans')

    sns.barplot(x='RSquaredMeans', y='Model', data=cv_final_train)
    plt.title("Cross Validation R2 Result")
    plt.show()


random_seed = 191
data = load_and_check_data('kc_house_data.csv')
X_train, X_test, y_train, y_test = preprocess_data(data, random_seed)
cv = KFold(n_splits = 5, random_state= random_seed)
pipelines = []
pipelines.append(
                ("Linear Regression", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Linear Regression", LinearRegression())
                      ]))
                )

pipelines.append(
                ("Random Forest", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Random Forest", RandomForestRegressor(random_state = random_seed))
                      ]))
                )

pipelines.append(
                ("Gradient Boosting", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Gradient Boosting", GradientBoostingRegressor(random_state = random_seed))
                      ]))
                )

pipelines.append(
                ("Support Vector Regressor", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Support Vector Regressor", SVR(kernel = 'linear'))
                      ]))
                )


pipelines.append(
                ("Bagging Regressor",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Bagging Regressor", BaggingRegressor(random_state=random_seed))
                 ]))) 

baseline_comparison(data, pipelines, cv)

final_model = RandomForestRegressor()
final_model.fit(X_train, y_train)
r2 = final_model.score(X_test, y_test)
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(r2)
print(rmse)
