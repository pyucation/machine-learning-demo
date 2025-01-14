import streamlit as st
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""
# Classification of Spotfiy Song Data
This exercise is about classification. You will use different techniques to classify the songs of three very famous bands/artists.
""")


img_bob_dylan = Image.open("figures/bob_dylan.jpg")
img_kiss = Image.open("figures/kiss.jpg")
img_beatles = Image.open("figures/beatles.jpg")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(img_bob_dylan, caption="Bob Dylan")

with col2:
    st.image(img_kiss, caption="Kiss")

with col3:
    st.image(img_beatles, caption="Beatles")


data = pd.read_csv("datasets/database_songs.csv")


st.markdown("""
### The Dataset
First step, when working with data, is to get to know the dataset. In the section below, you will find all the different features. A detailed explanation can be found on the [Spotify Website](https://developer.spotify.com/documentation/web-api/reference/). After reading this, have a look at the value ranges of the first five samples:
""")

data = data.drop(columns=["explicit"])
st.dataframe(data.head())

st.markdown("""
Before we start working, we need to take care of a small compatibility problem. Our labels are strings (e.g. "KISS"). But for some functions (like some of the plots we are doing later) we need the labels to be a number (integer). Replacing the class strings by a number results in: Bob Dylan -> 0, KISS -> 1, The Beatles -> 2
""")

# encoding variable of interest
y = data.artist_name

# Create an instance of the label encoder
# Label encoder encodes target labels with value between 0 and number_classes-1
# e.g.: [0, 4, 10, 10, 4, 20] --> [0, 1, 2, 2, 1, 3]
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Using a pandas dataframe to show the encoding
# encoder.classes_ contains the name of the labels
# encoder.transform() applies the encoding
all_features = data.drop(columns=["artist_name", "name"])

st.dataframe(pd.DataFrame(data=encoder.transform(encoder.classes_), index=encoder.classes_, columns=["Coding"]))


st.markdown("""
In the end, we want to keep two features to classify the songs. Select features and quantitatively compare how well they can be used for the classification. You will see three plots:
- Scatter Plot, showing the data points with their corresponding artists
- Kernel Density Estimate (KDE) Plot, showing a continuous probability density function for the selected features
""")

# select Feature 1
feature1 = st.selectbox("Feature 1", all_features.columns, index=list(all_features.columns).index("energy"))

# dynamically update Feature 2 options based on Feature 1 selection
feature2_options = all_features.columns[all_features.columns != feature1].tolist()
feature2_default = "Length (ln s)" if "Length (ln s)" in feature2_options else feature2_options[0]
feature2 = st.selectbox("Feature 2", feature2_options, index=feature2_options.index(feature2_default))


scatter_fig = px.scatter(all_features, x=feature1, y=feature2, color=data.artist_name)
scatter_fig.update_layout(title="Scatter Plot - Both Features")

# KDE Plot - Feature 1
kde_fig1 = px.histogram(all_features, x=feature1, color=data.artist_name, marginal="box")
kde_fig1.update_layout(title="KDE Plot - Feature 1")

# KDE Plot - Feature 2
kde_fig2 = px.histogram(all_features, x=feature2, color=data.artist_name, marginal="box")
kde_fig2.update_layout(title="KDE Plot - Feature 2")

st.plotly_chart(scatter_fig)
st.plotly_chart(kde_fig1)
st.plotly_chart(kde_fig2)


st.markdown("""
In the next plot you should be able to observe that the data is slightly imbalanced. KISS songs occur not as many times as songs by Bob Dylan or The Beatles. Therefore, we won't be using KISS songs in the following binary classification task. Later, when we do the multiclass classification, we will use all songs.
""")

hist_fig = px.histogram(y, nbins=3)
hist_fig.update_layout(
    title="Distribution of Artists",
    xaxis_title="Artist Name",
    yaxis_title="Frequency"
)

st.plotly_chart(hist_fig)


binary_data = data[data["artist_name"] != "KISS"]
y_binary = binary_data.artist_name
binary_data = binary_data.drop(columns=["artist_name", "name"])

encoder = LabelEncoder()
y_binary_encoded = encoder.fit_transform(y_binary)

X = binary_data[[feature1, feature2]]
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_encoded, test_size=0.3, random_state=350)


st.markdown("""
After splitting the data into train and test set, we need to preprocess the data. Here, we applied a StandardScaler and fitted only the training data to it. This is important since we assume that we do not know anything about the test data. With the StandardScaler fitted to the training data, we can also transform the test data. In other words: we use the mean and standard deviation of the training set for both training and test set. This is common practice.
""")

scaler = StandardScaler()

# replacing the values in our X_train by the normalized ones
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# replacing the values in our X_test by the normalized ones
X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# display the first few rows of X_train
st.write("Normalized Training Data:")
st.dataframe(X_train.head())


st.markdown("""
Let's have another look at the preprocessed training data.
""")

scatter_train_fig = px.scatter(X_train, x=feature1, y=feature2, color=encoder.inverse_transform(y_train))
scatter_train_fig.update_layout(title="Training Set")

st.plotly_chart(scatter_train_fig)

st.markdown("---")


# Task 1 - Binary Classification
st.markdown("""
### Task 1: Binary Classification
We have prepared our data for the binary classification task. Let's start with a simple logistic regression. You can select the parameters for logistic regression. Also, observe how the accuracy changes when you select different features.
""")

penalty = st.selectbox("Regularization", ["l2", "none"], index=0)

clf = LogisticRegression(penalty=penalty, random_state=42)
clf.fit(X_train, y_train)


x_min, x_max = X_test[feature1].min() - 1, X_test[feature1].max() + 1
y_min, y_max = X_test[feature2].min() - 1, X_test[feature2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

# we need to use a meshgrid in plotly as there is no direct
# equivalent to 'plot_decision_regions(X_test.values, y_test, clf=clf, legend=2)'
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

decision_boundary_fig = go.Figure(data=[
    go.Contour(x=np.linspace(x_min, x_max, 400), 
               y=np.linspace(y_min, y_max, 400), 
               z=Z, 
               colorscale='RdBu', 
               showscale=False)
])

for class_value in np.unique(y_test):
    decision_boundary_fig.add_trace(go.Scatter(
        x=X_test[feature1][y_test == class_value], 
        y=X_test[feature2][y_test == class_value],
        mode='markers',
        marker_symbol='circle',
        marker_line_color='black',
        marker_line_width=1,
        marker_size=10,
        name=str(encoder.inverse_transform(y_train)[class_value])
    ))

decision_boundary_fig.update_layout(
    title="Test Set Decision Boundary",
    xaxis_title=feature1,
    yaxis_title=feature2
)

st.plotly_chart(decision_boundary_fig)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(f"#### Accuracy: {accuracy:.2f}")
