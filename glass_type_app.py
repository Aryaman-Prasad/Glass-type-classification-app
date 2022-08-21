# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


features_columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()

st.title('Glass Type Prediction Web App')
st.sidebar.title('Glass Type Prediction Web App')

if st.sidebar.checkbox('Show Raw Data') :
  st.subheader('Glass Type Data Set')
  st.dataframe(glass_df)



st.sidebar.subheader('Visualization Selecter')
graphs = st.sidebar.multiselect('Select charts', ('Correlation Heatmap', 'Line Chart', 'Count Plot', 'Pie Chart', 'Area Chart', 'Box Plot'))
if 'Line Chart' in graphs :
  st.subheader('Line Chart')
  st.line_chart(glass_df)
if 'Area Chart' in graphs :
  st.subheader('Area Chart')
  st.area_chart(glass_df)
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'Correlation Heatmap' in graphs :
  st.subheader('Correlation Heatmap')
  plt.figure(figsize = (20, 20))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()
if 'Pie Chart' in graphs :
  st.subheader('Pie Chart')
  plt.figure(figsize = (20, 20))
  plt.pie(glass_df['GlassType'].value_counts(), labels = glass_df['GlassType'].value_counts().index, autopct = '%1.2f%%', startangle = 0, explode = np.linspace(0.05, 0.1, 6))
  st.pyplot()
if 'Count Plot' in graphs :
  st.subheader('Count Plot')
  plt.figure(figsize = (20, 20))
  sns.countplot(x = 'GlassType', data = glass_df)
  st.pyplot()
if 'Box Plot' in graphs :
  st.subheader('Box Plot')
  column_name = st.selectbox('Select the column', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize = (20, 20))
  sns.boxplot(glass_df[column_name])
  st.pyplot()

st.sidebar.subheader('Select values :')
ri = st.sidebar.slider('RI', float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider('Na', float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider('Mg', float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider('Al', float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider('Si', float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider('K', float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider('Ca', float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Ba', float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Fe', float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier :')
model = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

from sklearn.metrics import plot_confusion_matrix
if model == 'Support Vector Machine' :
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input('Error Rate', 1, 100, step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_value = st.sidebar.number_input('Gamma Value', 1, 100, step = 1)
  if st.sidebar.button('Classify') :
    st.subheader('Support Vector Machine')
    svc_model = SVC(kernel = kernel_input, C = c_value, gamma = gamma_value)
    svc_model.fit(X_train, y_train)
    y_predict = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The type of glass predicted is :", glass_type)
    st.write('Accuracy of model is :', accuracy)
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()
if model == 'Random Forest Classifier' :
  st.sidebar.subheader('Model Hyperparameters')
  n_estimators_input = st.sidebar.number_input('No. of trees in the forest', 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input('Maximum depth of the tree', 1, 100, step = 1)
  if st.sidebar.button('Classify') :
    st.subheader('Random Forest Classifier')
    rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
    rf_clf.fit(X_train, y_train)
    accuracy = rf_clf.score(X_test, y_test)
    glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is :', glass_type)
    st.write('Accuracy of model is :', accuracy)
    plot_confusion_matrix(rf_clf, X_test, y_test)
    st.pyplot()
if model == 'Logistic Regression' :
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input('C value', 1, 100, step = 1)
  max_iterations = st.sidebar.number_input('Maximum iterations', 10, 1000, step = 10)
  if st.sidebar.button('Classify') :
    st.subheader('Logistic Regression')
    log_reg = LogisticRegression(C = c_value, max_iter = max_iterations)
    log_reg.fit(X_train, y_train)
    accuracy = log_reg.score(X_test, y_test)
    glasstype = prediction(log_reg, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass predicted is', glasstype)
    st.write('Accuracy is :', accuracy)
    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()