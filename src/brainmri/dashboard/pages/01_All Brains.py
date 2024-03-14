import os
import pickle
from typing import List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from brainmri.constants import IMAGES_PATH, brain_image_paths
from brainmri.core.mri_image import MRIImage


def get_quadratic_fit(x, y):
    x = np.array(x)
    y = np.array(y)
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=st.session_state.degree)),
            ("linear", LinearRegression(fit_intercept=False)),
        ]
    )

    model = model.fit(x[:, np.newaxis], y)
    y_pred = model.predict(x[:, np.newaxis])

    # sort the values of x before line plot
    idx = np.argsort(x)
    x = x[idx]
    y_pred = y_pred[idx]
    return x, y_pred


if "mris" not in st.session_state:
    with st.spinner("Loading MRI Images..."):
        if "mris.pkl" in os.listdir():
            mris = pickle.load(open("mris.pkl", "rb"))
        else:
            mris = [MRIImage.from_file(path) for path in brain_image_paths]
            pickle.dump(mris, open("mris.pkl", "wb"))
        st.session_state["mris"] = mris

mris: List[MRIImage] = st.session_state.mris


cols = st.columns([1, 1.6], gap="medium")
with cols[0]:
    st.write("## Brain Volume and Head Circumference Analysis")
    st.write(
        """
This page is dedicated to the analysis of brain volumes and head circumferences of multiple MRI images. The right histogram shows the
distribution of gestational ages (GA) of the images. 
"""
    )
    st.write(
        """
Below are plots that show how brain volume and head circumference change with GA. The red line is a polynomial linear regression fit to the data.
The degree of the polynomial can be adjusted using the slider.
"""
    )
with cols[1]:
    import plotly.graph_objs as go

    gas = []
    for mri_image in mris:
        ga = mri_image.gestational_age()
        if ga is None or ga < 10:
            continue
        gas.append(ga)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=gas, histnorm="", nbinsx=30))
    fig.update_layout(
        title_text="Distribution of Gestational Ages",
        xaxis_title_text="Gestational Age (weeks)",
        yaxis_title_text="Frequency",
        bargap=0.2,
        bargroupgap=0.1,
    )
    fig.update_xaxes(range=[0, 45])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

cols = st.columns([1, 1], gap="medium")
with cols[0]:
    st.slider("Linear Regression Model Degree", 1, 10, key="degree", value=4)

st.write("### Brain Volume vs Gestational Age")


cols = st.columns([1, 1], gap="medium")

with cols[0]:
    with st.spinner("Calculating Brain Volumes..."):
        gas = []
        volumes = []
        for mri in mris:
            ga = mri.gestational_age()
            if ga != 0 and ga:
                volume = mri.volume_ml()
                gas.append(ga)
                volumes.append(volume)

        fig = px.scatter(x=gas, y=volumes, title="Total Volume vs Gestational Age")
        x, y_pred = get_quadratic_fit(gas, volumes)
        fig.add_scatter(
            x=x, y=y_pred, name="Regression Line", line=dict(color="red", width=2)
        )
        st.plotly_chart(fig)

with cols[1]:
    st.multiselect(
        "Intensity", list(range(1, 19)), key="intensities", default=[1, 8, 5]
    )

    fig = make_subplots()
    with st.spinner("Calculating Brain Volumes..."):
        for intensity in st.session_state.intensities:
            gas = []
            volumes = []
            for mri in mris:
                ga = mri.gestational_age()
                if ga != 0 and ga:
                    volume = mri.volume_ml(intensity)
                    gas.append(ga)
                    volumes.append(volume)

            x, y = get_quadratic_fit(gas, volumes)
            trace = go.Scatter(
                x=x,
                y=y,
                name=f"Intensity = {intensity}",
                mode="lines",
            )
            fig.add_trace(trace)
            trace = go.Scatter(
                x=gas,
                y=volumes,
                name=f"Intensity = {intensity}",
                mode="markers",
            )
            fig.add_trace(trace)

        st.plotly_chart(fig)

st.write("### Head Circumference vs Gestational Age")

if "max_cs" not in st.session_state or st.session_state["max_cs"] == []:
    st.session_state["max_cs"] = []
    st.session_state["gas"] = []
    st.write("Loading Brain Volumes")
    for mri_image in mris:
        max_c = mri_image.head_circumference()
        ga = mri_image.gestational_age()
        if max_c and ga:
            st.session_state["max_cs"].append(max_c)
            st.session_state["gas"].append(ga)
else:
    cols = st.columns([2, 1, 1], gap="medium")
    with cols[0]:
        fig = px.scatter(
            x=st.session_state["gas"],
            y=st.session_state["max_cs"],
            title="Head Circumference vs Gestational Age",
        )
        x, y_pred = get_quadratic_fit(
            st.session_state["gas"], st.session_state["max_cs"]
        )
        fig.add_scatter(
            x=x, y=y_pred, name="Regression Line", line=dict(color="red", width=2)
        )
        st.plotly_chart(fig)
    with cols[1]:

        st.image(str(IMAGES_PATH / "circum.png"))


footer = """<style>
.footer > a:link , .footer > a:visited{
color: black;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
opacity: 0.8;
color: black;
padding-left: 20px;
padding-top: 20px;
display: flex;
}

</style>
<div class="footer">
<p>Developed with 🖤 by <a style='display: block;' href="https://www.marc-julian.de/" target="_blank">Marc Julian Schwarz</a></p>
</div>
"""
st.sidebar.markdown(footer, unsafe_allow_html=True)
