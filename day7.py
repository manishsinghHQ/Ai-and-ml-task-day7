from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=load_breast_cancer()
x=pd.DataFrame(dataset.data)
y=pd.DataFrame(dataset.target)

x_vis = x.iloc[:, :2]  


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
cross_validation=cross_val_score(model,x,y,cv=5)
print(f"cross validation score for SVC linear is {cross_validation}")
cm=confusion_matrix(y_test,y_pred)
print(f"confusion matrix for SVC linear is \n {cm}")
model2=SVC(kernel='rbf')
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)
cm2=confusion_matrix(y_test,y_pred2)
cross_val_score2=cross_val_score(model2,x,y,cv=3)
print(f"cross validation score for SVC rbf is {cross_val_score2}")
print(f"confusion matrix for SVC rbf is \n {cm2}")
# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf'],
    
}

    
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(x_vis, y)
print("Best parameters from GridSearchCV:", grid.best_params_)
print("Best cross-validation score from GridSearchCV:", grid.best_score_)
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


# Plot using best model
plot_decision_boundary(grid.best_estimator_, x_vis, y)
