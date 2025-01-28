import numpy as np
import pandas as pd

import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import optuna
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
data = pd.read_csv('dataset1.csv')
y = data['RADIATION'].copy()
X = data.drop('RADIATION', axis=1).copy()
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=200)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
def get_model_rmse(params):
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')], early_stopping_rounds=10, verbose_eval=0)
    results = model.eval(dval)
    rmse = np.float64(re.search(r'[\d.]+$', results).group(0))
    return rmse
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.00001, 10.0)
    max_depth = trial.suggest_int('max_depth', 4, 8)
    l1_reg = trial.suggest_loguniform('l1_reg', 0.00001, 10.0)
    l2_reg = trial.suggest_loguniform('l2_reg', 0.00001, 10.0)
    
    params = {'learning_rate': learning_rate, 'max_depth': max_depth, 'alpha': l1_reg, 'lambda': l2_reg}
    
    return get_model_rmse(params)
study = optuna.create_study()
study.optimize(objective, n_trials=100, show_progress_bar=True)
best_params = study.best_params
best_params
model = xgb.train(best_params, dtrain, num_boost_round=10000, evals=[(dval, 'eval')], early_stopping_rounds=10)
y_true = np.array(y_test, dtype=np.float64)
y_pred = np.array(model.predict(dtest), dtype=np.float64)
r2 = r2_score(y_test, y_pred)

print("R^2 Score: {:.4f}".format(r2))
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)



# Assuming you have your predicted values (y_pred) and original values (y_test)
plt.scatter(y_test, y_pred)  # Scatter plot for actual vs predicted values

# Add labels and title
plt.xlabel("Original Value")
plt.ylabel("Predicted Value")
plt.title("Predicted vs. Original Values")

# Add a diagonal line for perfect prediction (optional)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')

# Show the plot
plt.grid(True)
plt.show()

