import streamlit as st


st.set_page_config(
    page_title="AICP - Welcome",
    page_icon="👋",
)

st.write("# Welcome to the TUM AI Competence Center")
st.write("## Linear Regression")
st.sidebar.success("Select a Task above.")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("./imgs/lin_reg.png", width=300)

st.markdown(
    """
    This tutorial, as the title suggests, is about linear regression: a key to unlocking predictive analytics. Linear regression stands as one of the most fundamental and widely used statistical techniques in data analysis. At its core, linear regression is a method used to model the relationship between a dependent variable and one or more independent variables. This powerful tool allows us to understand and quantify the impact of various factors on a given outcome, making it indispensable across numerous fields.

    ### Why is Linear Regression Important?
    1. **Simplicity and Interpretability**: Linear regression models are not only straightforward to implement but also easy to interpret, making them a go-to tool for gaining actionable insights from data.
    2. **Foundation for Other Techniques**: Understanding linear regression lays the groundwork for comprehending more complex models. Many advanced predictive models are built upon concepts foundational to linear regression.
    3. **Wide Range of Applications**: From economics to engineering, linear regression helps in predicting trends, determining strength of predictors, forecasting an effect, and many other critical analyses.

    ### Real-World Examples
    - **Economics** 🌍: Economists use linear regression to predict future economic growth by examining the relationships between economic indicators such as GDP, interest rates, and employment levels.
    - **Healthcare** 🏥: Linear regression aids in understanding the impact of lifestyle choices on health outcomes. For example, it might be used to analyze the effect of smoking on life expectancy.
    - **Business** 💼: In the business realm, linear regression can be employed to predict sales based on seasonal trends, or understand how changes in pricing affect product demand.

    ### In this Notebook
    ... you will dive deep into the theory and practical implementation of linear regression. You'll learn how to interpret and evaluate linear regression models using real-world datasets. Whether you're a beginner or looking to refresh your knowledge, this notebook is designed to provide a comprehensive understanding of linear regression in a practical, hands-on manner.

    Let's embark on this analytical journey to explore the impact and nuances of linear regression in data science!
    """ 
)