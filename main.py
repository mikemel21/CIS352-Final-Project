import streamlit as st
import pandas as pd
import inspect
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline # For setting up pipeline

# title
st.title("CIS 335 Project")

# import dataset
df = pd.read_csv("stroke-dataset.csv")

# remove ID column
df = df.drop(['id'], axis=1)

# Replace all N/A values in BMI column with the mean of that column
df['bmi'] = df['bmi'].fillna(df['bmi'].mean()).round(1)


# some data pre processing
df['gender'] = df['gender'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['ever_married'] = df['ever_married'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['work_type'] = df['work_type'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['Residence_type'] = df['Residence_type'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df['smoking_status'] = df['smoking_status'].apply(lambda x: x.lower() if isinstance(x, str) else x)

df['gender'].replace(['male', 'female', 'other'], [0, 1, 2], inplace=True)
df['ever_married'].replace(['yes', 'no'], [0, 1], inplace=True)
df['work_type'].replace(['children', 'never_work', 'never_worked', 'private', 'self-employed', 'govt_job'], [0, 1, 1, 2, 3, 4], inplace=True)
df['Residence_type'].replace(['urban', 'rural'], [0, 1], inplace=True)
df['smoking_status'].replace(['unknown', 'never smoked', 'formerly smoked', 'smokes'], [0, 1, 2, 3], inplace=True)
df = df.astype({'age': int})


def add_parameter_ui(classifier_name):
    pass


def get_classifier(classifier_name):
    match classifier_name:
        case "Random Forest":
            return RandomForestClassifier()
        case "AdaBoost":
            return AdaBoostClassifier()
        case "SVM":
            return SVC()
        case "Decision Tree":
            return DecisionTreeClassifier()


def get_normalizer(normalization_name):
    match normalization_name:
        case "MinMax":
            return MinMaxScaler()
        case "Z-Score":
            return StandardScaler()
        case "Power Transform":
            return PowerTransformer()
        case "None":
            return None


st.write(df)

rd = RandomForestClassifier()

docString = rd.__doc__

docString = docString.split(sep='\n')

for idx, line in enumerate(docString):
    if "Parameter" in line:
        parameterLine = idx
        st.write(line)

    if "Attribute" in line:
        attributeLine = idx
        st.write(line)

paramSection = docString[parameterLine-1:attributeLine]

st.write(paramSection)

with st.expander("**Select Classifier**"):
    classifier_name = st.selectbox(label="**Classifier options**",
                                           options=["Random Forest", "AdaBoost", "SVM", "Decision Tree"])

    classifier = get_classifier(classifier_name)

    st.write("Set Classifier Parameters")

    classifierParamNames = classifier.get_params()

    st.write()

    # this gets the parameters and default values for each one
    classifierParValues = []
    for par, defVal in classifier.get_params().items():
        tempVal = st.text_input(label=f"**{par}**", placeholder=defVal)

        # TODO: add description of parameter from docstring here in st.write()

        # if nothing has been put in this field use default
        if tempVal == "":
            classifierParValues.append(defVal)

        else:
            classifierParValues.append(tempVal)


with st.expander("Select Normalization Technique"):
    normalization_name = st.selectbox(label="Normalization Options",
                                              options=["None", "MinMax", "Z-Score", "Power Transform"])

    normalizer = get_normalizer(normalization_name)

    if normalizer is not None:
        st.write("Set Normalization Parameters")

        # this gets the parameters and default values for each one
        normalizerParValues = []
        for par, defVal in normalizer.get_params().items():
            tempVal = st.text_input(label=par, placeholder=defVal)

            # TODO: add description of parameter from docstring here in st.write()

            # if nothing has been put in this field use default
            if tempVal == "":
                normalizerParValues.append(defVal)

            else:
                normalizerParValues.append(tempVal)

    else:
        st.write("No normalization selected")


# pipe = Pipeline([
#     ('scaler', scaler),
#     ('classifier', classifier)])

