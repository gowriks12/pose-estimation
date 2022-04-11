from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("poselandmarks.csv")

df.head(5)
labels = df["class"].tolist()

# Train test split
X = df.drop(columns = ['class'])
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 0)

rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf_pred = rbf.predict(X_test)
accuracy_rbf = rbf.score(X_test, y_test)
# print(accuracy_rbf)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
linear_pred = linear.predict(X_test)
accuracy_lin = linear.score(X_test, y_test)

sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig_pred = sig.predict(X_test)
accuracy_sig = sig.score(X_test, y_test)

poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly_pred = poly.predict(X_test)
accuracy_poly = poly.score(X_test, y_test)

print("RBF",accuracy_rbf)
print("Linear",accuracy_lin)
print("Sigmoid",accuracy_sig)
print("Polynomial",accuracy_poly)


def create_svm_model(file):
    df = pd.read_csv(file)
    features = df.columns
    labels = df["class"].tolist()

    # Train test split
    X = df.drop(columns=['class'])
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)


    # poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    # # poly_pred = poly.predict(X_test)
    # accuracy_poly = poly.score(X_test, y_test)
    # print("Accuracy:", accuracy_poly)

    # Model linear kernel is used since it gives the best comparable accuracy
    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    linear_pred = linear.predict(X_test)
    accuracy_lin = linear.score(X_test, y_test)
    print("Accuracy:", accuracy_lin)

    return linear


svm_model = create_svm_model("poselandmarks.csv")
pickle.dump(svm_model, open('svm_pickle_file.sav', 'wb'))

# test = [0, 40.11234224026316, 78.89233169326408, 102.82509421342633, 133.60014970051492, 128.41339493993607, 188.76705220985997, 224.5373020235168, 254.03149411047443, 121.0495766204905, 193.8891435846783, 234.19649869287116, 267.1722290957651, 106.07544484940895, 148.08443537387717, 109.04127658827184, 79.37883848986453, 88.60022573334675, 99.72462083156798, 66.48308055437865, 42.485291572496]
# test1 = [0, 60.83584469702052, 118.06777714516353, 162.48384535085327, 207.03864373589778, 170.07351351694948, 227.343352662883, 258.30408436569485, 291.2387336876742, 173.9798838946618, 237.08437316702253, 274.23530042647684, 310.871356030111, 165.0272704737008, 227.17834403833479, 268.8215021161812, 306.28254929068356, 151.00993344810135, 203.17972339778396, 235.53555994796199, 269.3120866207085]
# test2 = [0, 47.41307836451879, 106.00471687618433, 154.434452114805, 191.12822920751398, 158.3729774930054, 153.3883959105121, 114.12712210513327, 121.49074038789952, 147.12239802287073, 144.183910336764, 94.72592042308166, 108.26818553942798, 135.6244815658294, 135.29966740535616, 94.76286192385707, 97.18538984847466, 128.89142717807107, 131.89768762188365, 101.84301645179212, 94.92101980067429]
# load the model from disk
# loaded_model = pickle.load(open('svm_pickle_file.sav', 'rb'))
# result = loaded_model.predict([test1])
# print(result[0])
