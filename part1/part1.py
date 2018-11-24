from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import sys

if len(sys.argv) == 2:
    print("Using file " + sys.argv[1])
    file_to_use = '../' + sys.argv[1]
else:
    print("Please add training file as argument.")
    sys.exit(1)

print("Creating classifier...")
clf = MLPClassifier(solver='sgd', learning_rate_init=0.5, hidden_layer_sizes=(5,), verbose=True, momentum=0.3,
                    activation='logistic', n_iter_no_change=5000, max_iter=10000)
print(clf)

print("\nReading file.")
full_file = pd.read_csv(file_to_use)

column_count = len(full_file.columns)

X_cols = [i for i in range(0, column_count-1)]

X = pd.read_csv(file_to_use, usecols=X_cols)
df_countries = pd.read_csv(file_to_use, usecols=[column_count-1])

# print(X)
# print(df_countries)
print("\nEncoding data...")
X = X.replace(regex={r'Yes': 1, r'No': 0})

# print(X)

enc = OneHotEncoder(handle_unknown='ignore')
enc = enc.fit(df_countries)
encoded_countries = enc.transform(df_countries).toarray()

# print(encoded_countries)
print("Data encoded.")

X_train, X_test = X[:50], X[50:]
y_train, y_test = encoded_countries[:50], encoded_countries[50:]

print("\nTraining...")
clf.fit(X_train.as_matrix(), y_train)

print('\nTESTING')
print("Training score: %f" % clf.score(X_train.as_matrix(), y_train))
print("Testing score: %f" % clf.score(X_test.as_matrix(), y_test))

model_name = sys.argv[1][:-4]
print("\nSaving model as " + model_name + ".joblib")
joblib.dump(clf, model_name + '.joblib')
print("Model saved.")

