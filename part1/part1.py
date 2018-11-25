from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pandas as pd
import sys

def train_classifier(file_to_use):
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

    print("\nEncoding data...")
    X = X.replace(regex={r'Yes': 1, r'No': 0})

    enc = OneHotEncoder(handle_unknown='ignore')
    enc = enc.fit(df_countries)
    encoded_countries = enc.transform(df_countries).toarray()

    print("Data encoded.")

    print("\nTraining...")
    clf.fit(X.as_matrix(), encoded_countries)

    print("\nTraining score: %f" % clf.score(X.as_matrix(), encoded_countries))

    model_name = sys.argv[1][:-4]
    print("\nSaving model as " + model_name + ".joblib")
    joblib.dump(clf, open('../' + model_name + '.joblib', 'wb'))
    print("Model saved.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Using file " + sys.argv[1])
        file_to_use = '../' + sys.argv[1]
        train_classifier(file_to_use)
    else:
        print("Please add training file as argument.")
        sys.exit(1)
