import sys
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib


def ask_user():
    if len(sys.argv) == 2:
        file_to_use = str("../" + sys.argv[1])

        df = pd.read_csv(file_to_use)       #Reading given file

    else:
        print("Please give a file.")
        sys.exit(1)

    clf = MLPClassifier()

    clf = joblib.load(file_to_use[:-4] + ".joblib")     #Loading classifier trained on given file

    enc = OneHotEncoder(handle_unknown='ignore')
    column_count = len(df.columns)
    df_countries = pd.read_csv(file_to_use, usecols=[column_count-1])
    enc.fit_transform(df_countries).toarray()           #Encoding locations for decoding predicition

    answers = []
    count = 0
    for column in df.columns:
        if column == "Dream Location":
            continue
        else:
            print(column + "?")     #Asking questions

        answer = ""
        while answer != "y" and answer != "Y" and answer != "n" and answer != "N":
            answer = input("Please answer Y for yes, N for no: ")
            if answer == "Y" or answer == "y":
                answers.append(1)
            elif answer == "N" or answer == "n":
                answers.append(0)

        count = count + 1
        if count == column_count // 2:
            make_guess(answers, enc, clf, column_count)

    make_guess(answers, enc, clf, column_count)


def make_guess(answers, enc, clf, column_count):
    current_answers = answers.copy()
    diff = column_count - 1 - len(current_answers)

    if diff > 0:                #If  not all questions answered, floods list with 0's
        for i in range(diff):
            current_answers.append(0)

    to_predict = [current_answers]
    prediction = enc.inverse_transform(clf.predict(to_predict))[0][0]   #Makes prediction

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
                print("That's a shame.")
    else:
        print("\nI have no guess.")
        if diff > 0:
            print("Let's continue!\n")


if __name__ == "__main__":
    ask_user()
