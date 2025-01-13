import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


st.set_page_config(
    page_title="Linear Regression - Task 3",
    page_icon="🏠",
)

st.write("# Linear Regression - Task 3")
st.write("## Regularization")

st.markdown("""The following task is for you to wrap your head around a very important concept in machine learning - regularization and overfitting. In the plots below you can see data points with a non-linear relationship. Play around with the polynomial degree $M$ and the L2-regularization parameter $\lambda$ to find the best possible combination for the corresponding dataset. What happens if you increase/decrease the two parameters? Make sure you understand what the regularization paramater does and how it can help to prevent overfitting.""")

t = np.linspace(0, 5, 8)
_t2 = np.linspace(0, 5, 1000)
x_41 = np.sin(t * np.pi) + np.random.normal(0, 0.1, 8)
y_41 = np.sin(t * np.pi)


M1 = st.slider("M₁", min_value=0, max_value=7, value=1)
lam1 = st.selectbox("λ₁", [0.00000001, 0.1, 1, 10, 100, 1000, 10000], index=0)

poly1 = PolynomialFeatures(degree=M1, include_bias=True)
poly_features = poly1.fit_transform(x_41.reshape(-1, 1), y_41.reshape(-1, 1))
model1 = Ridge(alpha=lam1)
model1.fit(poly_features, y_41.reshape(-1, 1))
overfit_features = poly1.transform(np.sin(_t2 * np.pi).reshape(-1, 1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y_41, mode='markers', name='Data Points'))
fig.add_trace(go.Scatter(x=_t2, y=np.sin(_t2 * np.pi), mode='lines', name='True Function'))
fig.add_trace(go.Scatter(x=_t2, y=model1.predict(overfit_features).ravel(), mode='lines', name='Prediction'))
fig.update_layout(
    xaxis_title="x",
    yaxis_title="y",
    yaxis=dict(range=[-1.3, 1.3])
)

st.plotly_chart(fig)
