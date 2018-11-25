import sys
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib


def ask_user():
    if len(sys.argv) == 2:
        file_to_use = str("../" + sys.argv[1])
        print("File to use: " + file_to_use)

        df = pd.read_csv(file_to_use)

        # print(df)
    else:
        print("Please give a file.")
        sys.exit(1)

    clf = MLPClassifier()

    if file_to_use == "../trip.csv":
        clf = joblib.load("../part1/trip.joblib")
    elif file_to_use == "../trip2.csv":
        clf = joblib.load("../part1/trip2.joblib")

    # print(clf)

    enc = OneHotEncoder(handle_unknown='ignore')
    column_count = len(df.columns)
    df_countries = pd.read_csv(file_to_use, usecols=[column_count-1])
    enc.fit_transform(df_countries).toarray()

    answers = []

    # print(df.columns)

    for column in df.columns:
        if column == "Short Stay":
            print("Would you only be staying for a short amount of time?")
        elif column == "Penguins":
            print("Is your dream location inhabited by penguins?")
        elif column == "Longest rivers":
            print("Does your dream location contain some of the worlds longest rivers?")
        elif column == "Island":
            print("Is your dream destination an island?")
        elif column == "Seaside":
            print("Is your dream destination known for its beaches?")
        elif column == "Historical":
            print("Is your dream location known for its rich history?")
        elif column == "Speaking Spanish":
            print("Is the national language of your dream location Spanish?")
        elif column == "Food":
            print("Is your dream destination known for its exotic cuisine?")
        elif column == "In Europe":
            print("Is your dream destination located within Europe?")
        else:
            continue

        answer = ""
        while answer != "y" and answer != "Y" and answer != "n" and answer != "N":
            answer = input("Please answer Y for yes, N for no: ")
            if answer == "Y" or answer == "y":
                answers.append(1)
            elif answer == "N" or answer == "n":
                answers.append(0)

        # print(answers)
        if column == "Seaside":
            make_guess(answers, enc, clf, column_count)

    make_guess(answers, enc, clf, column_count)


def make_guess(answers, enc, clf, column_count):
    current_answers = answers.copy()
    diff = column_count - 1 - len(current_answers)

    if diff > 0:
        for i in range(diff):
            current_answers.append(0)

    to_predict = [current_answers]
    prediction = enc.inverse_transform(clf.predict(to_predict))[0][0]

    if prediction != None:
        print("I think your dream destination is " + prediction)
        is_correct = input("Am I correct? ")

        if is_correct == "y":
            print("Hooray!")
            sys.exit(1)
        elif is_correct == "n":
            if diff != 0:
                print("Oh dear. Let's continue...")
            else:
                print("Uh oh. I guess I don't know!")
    else:
        print("\nI have no guess.")
        if diff > 0:
            print("Let's continue!\n")


if __name__ == "__main__":
    ask_user()
