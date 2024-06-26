import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline # For setting up pipeline

# title
st.title("CIS 335 Project: Stroke Dataset")

# import dataset
df = pd.read_csv("stroke-dataset.csv")

# remove ID column
df = df.drop(["id"], axis=1)

# Replace all N/A values in BMI column with the mean of that column
df["bmi"] = df["bmi"].fillna(df["bmi"].mean()).round(1)


# some data pre processing
df["gender"] = df["gender"].apply(lambda x: x.lower() if isinstance(x, str) else x)
df["ever_married"] = df["ever_married"].apply(
    lambda x: x.lower() if isinstance(x, str) else x
)
df["work_type"] = df["work_type"].apply(
    lambda x: x.lower() if isinstance(x, str) else x
)
df["Residence_type"] = df["Residence_type"].apply(
    lambda x: x.lower() if isinstance(x, str) else x
)
df["smoking_status"] = df["smoking_status"].apply(
    lambda x: x.lower() if isinstance(x, str) else x
)

df["gender"].replace(["male", "female", "other"], [0, 1, 2], inplace=True)
df["ever_married"].replace(["yes", "no"], [0, 1], inplace=True)
df["work_type"].replace(
    ["children", "never_work", "never_worked", "private", "self-employed", "govt_job"],
    [0, 1, 1, 2, 3, 4],
    inplace=True,
)
df["Residence_type"].replace(["urban", "rural"], [0, 1], inplace=True)
df["smoking_status"].replace(
    ["unknown", "never smoked", "formerly smoked", "smokes"], [0, 1, 2, 3], inplace=True
)
df = df.astype({"age": int})


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "SVM":
        C = st.slider("C", 0.01, 10.0)
        randomState = st.slider("Random State", 1, 10)
        params["C"] = C
        params["random-state"] = randomState
    elif classifier_name == "Random Forest":
        criterion = st.selectbox(
            label="Select Criterion", options=["gini", "entropy", "log_loss"]
        )
        max_depth = st.slider("max depth", 2, 10)
        n_estimators = st.slider("n_estimators", 2, 50)
        randomState = st.slider("Random State", 1, 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["random-state"] = randomState
    elif classifier_name == "Decision Tree":
        max_depth = st.slider("C", 2, 10)
        randomState = st.slider("Random State", 1, 10)
        splitter = st.selectbox(label="Select Splitter", options=["best", "random"])
        criterion = st.selectbox(
            label="Select Crtiterion", options=["gini", "entropy", "log_loss"]
        )
        params["max_depth"] = max_depth
        params["random-state"] = randomState
        params["splitter"] = splitter
        params["criterion"] = criterion
    elif classifier_name == "AdaBoost":
        # estimator = st.selectbox(label="Select Estimator", options=["Decision Tree"] )
        n_estimators = st.slider("n estimators", 1, 100)
        randomState = st.slider("Random State", 1, 10)
        params["n_estimators"] = n_estimators
        params["random-state"] = randomState
    return params


def get_classifier(classifier_name, params):
    match classifier_name:
        case "Random Forest":
            return RandomForestClassifier(
                criterion=params["criterion"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                random_state=params["random-state"],
            )
        case "AdaBoost":
            return AdaBoostClassifier(
                n_estimators=params["n_estimators"], random_state=params["random-state"]
            )
        case "SVM":
            return SVC(C=params["C"], random_state=params["random-state"])
        case "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=params["max_depth"],
                splitter=params["splitter"],
                criterion=params["criterion"],
                random_state=params["random-state"],
            )


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

with st.expander("**Select Classifier**"):
    classifier_name = st.selectbox(
        label="**Classifier options**",
        options=["Random Forest", "AdaBoost", "SVM", "Decision Tree"],
    )

    st.write("Set Classifier Parameters")

    params = add_parameter_ui(classifier_name)
    classifier = get_classifier(classifier_name, params)


with st.expander("Select Normalization Technique"):
    normalization_name = st.selectbox(
        label="Normalization Options",
        options=["None", "MinMax", "Z-Score", "Power Transform"],
    )

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

x = df.iloc[:, :9]
y = df.iloc[:, 10]

smt = SMOTE(random_state=params.get("random-state"))
x, y = smt.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=params.get("random-state")
)

pipe = Pipeline([("scaler", normalizer), ("classifier", classifier)])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc = pipe.score(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write("Recall = ", recall)
st.write("Precision = ", precision)
st.write("F1 = ", f1)
st.write("Accuracy = ", acc)
st.header("Visualizations")
st.caption("Visualizations of each feature")

# Visualization of features
feature_options = df.columns.tolist()
selected_feature = st.selectbox("Select a feature:", feature_options)

# Create the bar plot visualization
fig, ax = plt.subplots(figsize=(8, 6))
if selected_feature in ['hypertension', 'ever_married', 'heart_disease']:
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(["No", "Yes"])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
elif selected_feature == 'Residence_type':
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(["Urban", "Rural"])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
elif selected_feature == 'smoking_status':
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(["unknown", "never smoked", "formerly smoked", "smokes"])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
elif selected_feature == 'gender':
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(['Male', 'Female', 'Other'])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
elif selected_feature == 'work_type':
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(["children", "never_work", "private", "self-employed", "govt_job"])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
elif selected_feature == 'stroke':
    value_counts = df[selected_feature].value_counts()
    ax.bar(value_counts.index, value_counts.values)
    ax.set_xticks(range(len(value_counts.index)))
    ax.set_xticklabels(["Stroke", "No Stroke"])
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bar Plot of {selected_feature}")
    st.pyplot(fig)
else:
    ax.hist(df[selected_feature], bins=20)
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {selected_feature}")
    st.pyplot(fig)
