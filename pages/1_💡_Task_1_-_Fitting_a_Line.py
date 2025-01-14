import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Linear Regression",
    page_icon="💡",
)

st.write("# Linear Regression - Task 1")
st.write("## Fitting a Line")

st.markdown("""In the plot below you can see some randomly generated data points as well as a linear fit of the data points using the linear regression. Change the slider value on the left to increase/decrease the number of data points. Observe how the linear fit changes.""")



def generate_data(n):
    x = np.random.normal(0, 1, size=n).reshape(-1, 1)
    y = np.random.normal(0, 1, size=n).reshape(-1, 1)
    return x, y

# use streamlits session state to keep generated data when user changes the outlier checkbox
if 'x' not in st.session_state or 'y' not in st.session_state:
    st.session_state['x'], st.session_state['y'] = generate_data(5)

points = st.slider("Number of data points", min_value=2, max_value=30, value=5)

# regenerate data if slider changes
if points != len(st.session_state['x']):
    st.session_state['x'], st.session_state['y'] = generate_data(points)


outlier = st.checkbox("Add Outlier")

# get data from session state
x, y = st.session_state['x'], st.session_state['y']

if outlier:
    outlier_value = (0.7, 5)
    x_with_outlier = np.vstack((x, np.array([outlier_value[0]]).reshape(-1, 1)))
    y_with_outlier = np.vstack((y, np.array([outlier_value[1]]).reshape(-1, 1)))
    reg = LinearRegression().fit(x_with_outlier, y_with_outlier)
    y_pred = reg.predict(x_with_outlier)
else:
    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x.ravel(), y=y.ravel(), mode='markers', name='data points'))
if outlier:
    fig.add_trace(go.Scatter(x=[outlier_value[0]], y=[outlier_value[1]], mode='markers', marker_color='green', name='outlier'))
fig.add_trace(go.Scatter(x=x.ravel(), y=y_pred.ravel(), mode='lines', line_color='red', name='linear fit'))
fig.update_layout(title="Linear Regression Fit", xaxis_title="x", yaxis_title="y", legend=dict(x=0, y=1))

st.plotly_chart(fig)
