from datenmanagement_wetter import x_train, x_test, x_vali, risk_mm_test, risk_mm_train, risk_mm_vali
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# PARAMS_RF = {"n_estimators": 10, "max_depth": 4, "min_samples_split": 100, "min_samples_leaf": 50}
PARAMS_GB = {"n_estimators": 20, "max_features": 3, "max_depth": 4}

regressor = GradientBoostingRegressor(**PARAMS_GB)
regressor.fit(X=x_train, y=risk_mm_train)

predictions_train = regressor.predict(x_train)
# print(predictions_train)
mse_train = mean_squared_error(risk_mm_train, predictions_train)
mae_train = mean_absolute_error(risk_mm_train, predictions_train)

print("MSE Train", mse_train)
print("MAE Train", mae_train)

predictions_vali = regressor.predict(x_vali)
mse_vali = mean_squared_error(risk_mm_vali, predictions_vali)
mae_vali = mean_absolute_error(risk_mm_vali, predictions_vali)

print("MSE Vali", mse_vali)
print("MAE Vali", mae_vali)
print("Overfitting: ", mse_vali - mse_train)
