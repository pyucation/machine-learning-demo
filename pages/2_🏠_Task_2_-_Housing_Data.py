import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Linear Regression - Task 2",
    page_icon="🏠",
)

st.write("# Linear Regression - Task 2")
st.write("## Housing Data")

st.markdown("""You have seen the famous Housing Price Dataset. The goal is to predict the price of a house based on some features like the number of bedrooms, the area, and so on.
            Now you get the chance to play around with this dataset. The goal of this task is to find the optimal solution to a regression problem by adjusting the paramaters of the model.
To make it a bit easier, you only have to use the feature *area*. Calculate $\mathbf{w}^*$ (in German: Anstieg $w_1$ und Schnittpunkt mit y-Achse $w_0$) for the first eight data points of the dataset and then select your $w_0$ and $w_1$. "Show optimal solution" displays the linear fit for the whole (training) data set (based on the feature area).""")
st.markdown("""*Hint: Closely look at the y-axis and observe that the price displayed is divided by 1000.*""")


data = pd.read_csv("datasets/Housing.csv")
# shuffle data
data = data.sample(frac=1.0, random_state=42)
train = data[50:]
test = data[:50]
st.dataframe(data.head())

w0 = st.number_input("Intercept w₀ (x 1000)", min_value=-5000000, max_value=5000000, value=0) * 1000
w1 = st.number_input("Slope w₁", min_value=-500, max_value=2000, value=0)
opt_solution = st.checkbox("Show optimal solution")


# solution
# solution, based on 8 data points
# X = train["area"][:8].to_numpy()
# X = np.vstack((np.ones_like(X), X))
# X = X.T

# y_ = train["price"][:8].to_numpy()

# # w0, w1
# w = np.linalg.inv(X.T @ X) @ X.T @ y_
# # [-221298.89433326    1008.99695967]
# # print(w)

# _x = np.linspace(2000, 13000, 100)
# _y = w[0] + w[1] * _x
# # plt.scatter(train["area"], train["price"]/1000, label="all")
# # plt.plot(_x, _y/1000, label='Linear Fit', color='red')


_x = np.linspace(train["area"].min(), train["area"].max(), 100).reshape(-1, 1)

model = LinearRegression()
X_all = train["area"].to_numpy()
y_all = train["price"].to_numpy()

_ = model.fit(X_all.reshape(-1, 1), y_all.reshape(-1, 1))

fig = go.Figure()


fig.add_trace(go.Scatter(x=train["area"], y=train["price"]/1000, mode='markers', name='all data'))
fig.add_trace(go.Scatter(x=train["area"][:8], y=train["price"][:8]/1000, mode='markers', name='first 8'))
fig.add_trace(go.Scatter(x=_x.ravel(), y=(w0 + w1 * _x).ravel()/1000, mode='lines', name='your solution', line=dict(color='red')))

if opt_solution:
    fig.add_trace(go.Scatter(x=_x.ravel(), y=model.predict(_x).ravel()/1000, mode='lines', name='optimal solution', line=dict(color='green')))

fig.update_layout(
    title="Housing Data - Linear Regression",
    xaxis_title="area [feet²]",
    yaxis_title="price x $1000",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig)
