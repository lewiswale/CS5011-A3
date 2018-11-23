from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd

clf = MLPClassifier(solver='sgd', learning_rate_init=0.5, hidden_layer_sizes=(5,), verbose=True, momentum=0.3,
                    activation='logistic', n_iter_no_change=5000, max_iter=10000)

file_to_use = '../trip2.csv'

full_file = pd.read_csv(file_to_use)

column_count = len(full_file.columns)
print(column_count)

X_cols = [i for i in range(0, column_count-1)]

X = pd.read_csv(file_to_use, usecols=X_cols)
df_countries = pd.read_csv(file_to_use, usecols=[column_count-1])

print(X)
print(df_countries)

X = X.replace(regex={r'Yes': 1, r'No': 0})

print(X)

unique_countries = df_countries['Dream Location'].unique()

enc = OneHotEncoder(handle_unknown='ignore')
encoded_countries = enc.fit_transform(df_countries)

X_train, X_test = X[:50], X[50:]
y_train, y_test = encoded_countries[:50], encoded_countries[50:]


clf.fit(X_train.as_matrix(), y_train)

# joblib.dump(clf, 'travel_genie.joblib')

print('TESTING')
print("Training score: %f" % clf.score(X_train.as_matrix(), y_train))
print("Testing score: %f" % clf.score(X_test.as_matrix(), y_test))

