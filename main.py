import streamlit as st
import pandas as pd
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

# title
st.title("CIS 335 Project")

# import dataset
df = pd.read_csv("stroke-dataset.csv")

# remove ID column
df = df.drop(['id'], axis=1)

# Replace all N/A values in BMI column with the mean of that column
df['bmi'] = df['bmi'].fillna(df['bmi'].mean()).round(1)

classifier_name = st.sidebar.selectbox(label="Select classification", options=["Random Forest", "AdaBoost", "SVM", "Decision Tree"])

normalization_name = st.sidebar.selectbox(label="Select Normalization", options=["None", "MinMax", "Z-Score", "Power Transform"])


def add_parameter_ui(classifier_name):
    pass


def get_classifier(classifier_name, parameters):
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
        case "Min Max":
            return MinMaxScaler()
        case "Z-Score":
            return StandardScaler()
        case "Power Transform":
            return PowerTransformer()
        case "None":
            return "No Normalizer Selected"

st.write(df)

params = add_parameter_ui(classifier_name)
classifier = get_classifier(classifier_name, params)
normalizer = get_normalizer(normalization_name)

st.write(classifier)
st.write("\n\n")
st.write(normalizer)
