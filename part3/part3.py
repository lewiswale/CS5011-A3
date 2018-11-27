import sys
sys.path.insert(0, "../part1")
import part1
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

def ask_user():
    if len(sys.argv) == 2:
        file_to_use = str("../" + sys.argv[1])

        df = pd.read_csv(file_to_use)

        # print(df)
    else:
        print("Please give a file.")
        sys.exit(1)

    clf = MLPClassifier()

    clf = joblib.load(file_to_use[:-4] + ".joblib")

    # print(clf)

    enc = OneHotEncoder(handle_unknown='ignore')
    column_count = len(df.columns)
    df_countries = pd.read_csv(file_to_use, usecols=[column_count-1])
    enc.fit_transform(df_countries).toarray()

    answers = []
    count = 0
    for column in df.columns:
        if column == "Dream Location":
            continue
        else:
            print(column + "?")

        answer = ""
        while answer != "y" and answer != "Y" and answer != "n" and answer != "N":
            answer = input("Please answer Y for yes, N for no: ")
            if answer == "Y" or answer == "y":
                answers.append(1)
            elif answer == "N" or answer == "n":
                answers.append(0)

        count = count + 1
        if count == column_count // 2:
            make_guess(answers, enc, clf, column_count, file_to_use)

    make_guess(answers, enc, clf, column_count, file_to_use)


def make_guess(answers, enc, clf, column_count, file_to_use):
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
                update_data(answers, file_to_use, column_count)
    else:
        print("\nI have no guess.")
        if diff > 0:
            print("Let's continue!\n")
        else:
            update_data(answers, file_to_use, column_count)

def update_data(answers, file_to_use, column_count):
    dream_dest = input("Please tell me your dream destination: ")   #Gets correct answer
    yes = ""
    while yes != "y" and yes != "Y":
        yes = input("Your dream destination is " + dream_dest + " correct? (y/n): ")
        if yes == "n" or yes == "N":
            dream_dest = ""
            while dream_dest != "":
                dream_dest = input("Please tell me your dream destination: ")

    yes = ""
    new_feature = ""
    while yes != "n" and yes != "N":        #Gets defining feature of new location
        yes = input("Is there a defining feature to this destination that we have not asked about? (y/n) ")
        if yes == "Y" or yes == "y":
            new_feature = input("Please tell me this new feature!: ")
            yes = "n"

    decoded_answers = []        #Decodes users answers
    for answer in answers:
        if answer == 1:
            decoded_answers.append("Yes")
        elif answer == 0:
            decoded_answers.append("No")

    original_file = pd.read_csv(file_to_use)

    if new_feature != "":   #Appends yes if new feature given, builds new column in data
        new_col_values = []
        for i in range(len(original_file)):
            new_col_values.append("No")

        original_file.insert(loc=column_count-1, column=new_feature, value=new_col_values)
        decoded_answers.append("Yes")

    decoded_answers.append(dream_dest)
    print(decoded_answers)

    new_row = pd.DataFrame([decoded_answers], columns=original_file.columns)
    print("Adding new row to data:")
    print(decoded_answers)
    new_file = original_file.append(new_row, ignore_index=True)     #Adding new data entry
    print("Entry added.")
    update_additions(dream_dest, new_feature, file_to_use)  #Updating additions file
    new_file.to_csv(file_to_use, index=False)           #Overwriting original data file
    part1.train_classifier(file_to_use)             #Retraining classifier
    print("\nClassifier retrained...")

def update_additions(dream_dest, new_feature, file_to_use):
    additions_data = pd.read_csv("../additions.csv")
    new_entry = [dream_dest, "Location", file_to_use]
    new_row = pd.DataFrame([new_entry], columns=additions_data.columns)
    additions_data = additions_data.append(new_row, ignore_index=True)

    if new_feature != "":
        new_entry = [new_feature, "Feature", file_to_use]
        new_row = pd.DataFrame([new_entry], columns=additions_data.columns)
        additions_data = additions_data.append(new_row, ignore_index=True)

    additions_data.to_csv("../additions.csv", index=False)

if __name__ == "__main__":
    ask_user()
