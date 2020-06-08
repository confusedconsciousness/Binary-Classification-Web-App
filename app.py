import pandas as pd
import streamlit as st
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
# we don't want to load data, each time our app is loaded, this will lead to 
# huge computation if our dataset is big
@st.cache(persist=True)
def load_data(path2dataset):
    data = pd.read_csv(path2dataset)
    # since the dataset contains categorical variable we'll need to encode them
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache(persist=True)
def split(dataframe):
    y = dataframe['class']
    X = dataframe.drop(columns=['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def plot_metrics(metrics_list, model, X_test, y_test, class_names):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()
    
    if 'Precision Recall Curve' in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification")
    st.markdown("Are your Mushrooms safe to eat?")
    df = load_data("mushrooms.csv")
    X_train, X_test, y_train, y_test = split(df)

    class_names = ['edible', 'poisonous']

    st.sidebar.subheader("Choose Classifier")
    classifiers = st.sidebar.selectbox("Classifiers", ("Support Vector Machines", "Logistic Regression", "Random Forests"))

    if classifiers == "Support Vector Machines":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernels", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("Which Metric to Plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

        # we'll create a button that will run the model
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    
    if classifiers == "Logistic Regression":
        st.sidebar.subheader("Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("max_iter", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("Which Metric to Plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

        # we'll create a button that will run the model
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    if classifiers == "Random Forests":
        st.sidebar.subheader("Hyperparameters")
        n_estimators = st.sidebar.number_input("Total number of trees in the Forest.", 100, 5000, step=10, key='n_estimator')
        max_depth = st.sidebar.number_input("Maximum depth of the Trees", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap sample?", ("True", "False"), key='bootstrap')

        metrics = st.sidebar.multiselect("Which Metric to Plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

        # we'll create a button that will run the model
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics, model, X_test, y_test, class_names)

    # let's add a checkbox, to show the Raw data
    if st.sidebar.checkbox("Show Data", False):
        # if the user has checked it, then show the data on main page
        st.subheader("Mushrooms Dataset (Classification)")
        # let's write some description
        st.markdown("This dataset contains 22 features")
        st.write(df)
    

if __name__ == "__main__":
    main()