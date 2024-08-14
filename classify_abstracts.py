import pandas as pd
import numpy as np
import json
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CATEGORIES = {"RCT meta-analysis": "pool individual randomized controlled trials (RCTs) together to arrive at an overall estimate of the effect of the intervention under consideration", 
              "Systematic Review": "summary of research results (evidence) that uses explicit and reproducible methods to systematically search, critically appraise, and synthesize on a specific issue", 
              "Randomized Control Trial/RCT": "randomly assigns participants into an experimental group or a control group", 
              "Cohort Study": "recruit and follow participants who share a common characteristic, such as a particular occupation or demographic similarity.", 
              "Case-Controlled Study": "compares two groups of people: those with the disease or condition under study (cases) and a very similar group of people who do not have the disease or condition (controls)", 
              "Cross-Sectional Study": "observational studies that analyze data from a population at a single point in time", 
              "Cross-Sectional Survey": "observational surveys that analyze data from a population at a single point in time.", 
              "Case Report": "A detailed report of the diagnosis, treatment, and follow-up of an individual patient", 
              "Case Study": "a process or record of research in which detailed consideration is given to the development of a particular person, group, or situation over a period of time.", 
              "Observational Study": "research studies in which researchers collect information from participants or look at data that was already collected.", 
              "Observational Cohort Study": "a type of observational study that follows a group of participants over a period of time, examining how certain factors (like exposure to a given risk factor) affect their health outcomes.", 
              "Modeled Data": "analyzing and defining all the different data types your business collects and produces, as well as the relationships between those bits of data.", 
              "Exploratory Analyses": "analysis approach that identifies general patterns in the data", 
              "Editorial": "an article written by the senior editorial people or publisher of a newspaper, magazine, or any other written document, often unsigned.", 
              "Non-Systematic Review": "an informative, rather than all-encompassing, review of the literature on a topic.", 
              "Expert Opinion": "An expert's opinion", 
              "Unknown": "None of the other categories"}

def nn_classify(dataframe):
    # Split the data into input features (abstracts) and labels
    X = dataframe['Abstract']
    y = dataframe['Study Design']

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer to convert abstracts into numerical features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Convert labels to one-hot encoded vectors
    num_classes = len(label_encoder.classes_)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create a neural network model, use relu activation
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=-1))

    print(classification_report(y_test_labels, y_pred_labels))

def ml_classify(dataframe, type):
    # Split the data into input features (abstracts) and labels
    X = dataframe['Abstract']
    y = dataframe['Study Design']

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer to convert abstracts into numerical features
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create a Random Forest classifier
    if type == 2:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create a Support Vector Machine classifier
    elif type == 3:
        clf = SVC(kernel='rbf', random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Convert numerical predictions back to original labels
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # Evaluate the classifier
    print(classification_report(y_test_labels, y_pred_labels))
    print(accuracy_score(y_test_labels, y_pred_labels))

def get_vox_category(abstract, SEARCH_URL, token):
    if abstract == "nan":
        return "Unknown"
    try: 
        categories = str(CATEGORIES)
        query=f"""You are a research paper study design categorization expert. 
        Given a paper's abstract, your job is to decide which category the paper's study design falls under. 
        I will supply you a list of categories, each category will have its corresponding definition. \n
        Select exactly one category that fits the study design the best. \n
        If a field is missing, use the remaining fields to make your pick. \n
        IF THERE ARE MULTIPLE CATEGORIES THAT FIT, choose the category that appears first in the list! \n
        Provide the cateogory name exactly how its spelled in the cateogry list. \n
        Provide the category as a JSON with a single key spelled exactly: 'category', and no premable, explanation, or definition. \n
        Here is the abstract: \n\n {abstract} \n\n
        Here are the categories: {categories} \n
        """
        url = f"{SEARCH_URL}/chatCompletion"
        payload = json.dumps({
            "messages": [{"role": "user", "content": query}],
            "engine": "gpt-35-turbo",
            "max_tokens": "1400",
            "temperature": 0
        })
        headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
        }
        response = requests.post(url, headers=headers, data=payload)
        category = json.loads(json.loads(response.json()['result'])['content'])['category']
        
    except Exception as e:
        print("VOX Categorization error: ", e)
        print("Retrying...")
        return get_vox_category(abstract, SEARCH_URL, token)
    
    return str(category)

def vox_classify(dataframe, url, token):
    count = 0
    total = len(dataframe["Abstract"])

    for index, row in dataframe.iterrows():
        count += 1
        print("Categorizing Abstract #", count,"/", total, "...")
        abstract = row["Abstract"]
        category = get_vox_category(abstract, url, token)
        # Add the category to the "VOX SD" column
        dataframe.at[index, "VOX SD"] = category
    
    print("Categorization Complete!")

def federate_auth(CLIENT_ID, CLIENT_SECRET, PINGFEDERATE_URL):
    """Obtains auth access token for accessing GPT endpoints"""
    try:
        payload = f"client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(PINGFEDERATE_URL, headers=headers, data=payload)
        token = response.json()["access_token"]
        return token
    except requests.exceptions.HTTPError as e:
        print("HTTP Error:", e.response.status_code, e)
    except requests.exceptions.ConnectionError as e:
        print("Connection Error:", e.response.status_code, e)
    except requests.exceptions.Timeout as e:
        print("Timeout Error:", e.response.status_code, e)
    except requests.exceptions.RequestException as e:
        print("Other Error:", e.response.status_code, e)

def main():
    # Choose the classifier
    print("===================================")
    print("Choose classification technique:")
    print("1: VOX (LLM)")
    print("2: Randon Forest")
    print("3: SVMs")
    print("4: Neural Networks")
    selection = input("Make selection: ")
    if selection not in ["1", "2", "3", "4"]:
        print("Must be number from 1-5!")
        main()

    selection = int(selection)
    
    # Obtain the data set file path
    while(True):
        data_file_path = input("Enter file path of data set (ex: /dir/data set.xlsx): \n")
        if data_file_path == None:
            print("###Error! Cannot leave field empty")
        else:
            try:
                dataset_df = pd.read_excel(data_file_path)
                break
            except:
                print("###Error: Incorrect master format! Please be sure master excel sheet is NOT open and correctly spelled!")

    # VOX (LLM) clasification
    if selection == 1:
        with open("credentials.json", "r") as file:
            credentials = json.load(file)

        # Get the username and password from the credentials -- SECRET
        CLIENT_ID = credentials.get("username")
        CLIENT_SECRET = credentials.get("password")
        endpoint_url = credentials.get("endpoint_url")
        PINGFEDERATE_URL = credentials.get("pingfederate_url")
        api_token = federate_auth(CLIENT_ID, CLIENT_SECRET, PINGFEDERATE_URL)
        vox_classify(dataset_df, endpoint_url, api_token)

        dataset_df.to_excel(data_file_path)
        print("Labels added to", data_file_path)

    # Random Forest classification (ensemble DT) & SVM
    if selection == 2 or 3:
        ml_classify(dataset_df, selection)

    # Neural Network
    if selection == 4:
        nn_classify(dataset_df)

    main()

if __name__ == "__main__":
    main()