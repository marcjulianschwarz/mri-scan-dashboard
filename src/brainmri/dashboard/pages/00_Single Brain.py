import streamlit as st

from brainmri.constants import BRAIN_IMAGE_NAME_TO_PATH_MAPPING, IMAGES_PATH, PLOTS_PATH
from brainmri.core import MRIImage
from brainmri.nn import Net, predict_ga, predict_ga_reg
from brainmri.viz import plot_3d_brain, plot_animator, volume_histogram

if "mri_image" not in st.session_state:
    image_path = list(BRAIN_IMAGE_NAME_TO_PATH_MAPPING.values())[0]
    st.session_state["mri_image"] = MRIImage.from_file(image_path)

mri_image: MRIImage = st.session_state.mri_image


def set_mri_image():
    image_path = BRAIN_IMAGE_NAME_TO_PATH_MAPPING[st.session_state.img_name]
    st.session_state["mri_image"] = MRIImage.from_file(image_path)


st.write("## Single Brain Analysis")
cols = st.columns([1.4, 0.5, 0.7], gap="medium")
with cols[0]:
    st.write(
        """
This page allows the analysis of single brain MRI images. An image, axis and slice can be selected on the left. 
The selected slice will be displayed on the right. The total volume of the brain is calculated as a sum of
all slice volumes. The current slice volume can be seen as well.
"""
    )
    st.write(
        """
The head circumfrence is estimated by calculating the circumference of the brain at its widest point. It is additionally
estimated by 
"""
    )
    st.write("$$h=1.6\cdot (s_b + s_o)$$ ")
    st.write(
        """
where $s_b$ is the biparietal diameter and $s_o$ is the occipitofrontal diameter (see Figure C and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5504265/).
"""
    )
    st.write(
        """
The gestational age (GA) is estimated using a convolutional neural network trained on MRI images and their corresponding GA. For the test
set the model has a mean absolute error of about one week. The same was done using a linear regression model 
on the volumes and head circumferences. The regression model only performes slightly worse than the CNN. By using an ensemble of both models
the variance of the predictions could be reduced (better than any on the models alone).
"""
    )
    st.write(
        """ 
"""
    )
with cols[1]:
    st.image(str(IMAGES_PATH / "skull.png"))

with cols[2]:
    st.image(str(IMAGES_PATH / "circum.png"))

st.divider()

cols = st.columns([0.6, 0.5, 0.5, 0.7, 1], gap="medium")
with cols[0]:
    st.write("#### Select")

    st.selectbox(
        "Select MRI Image",
        BRAIN_IMAGE_NAME_TO_PATH_MAPPING.keys(),
        on_change=set_mri_image,
        key="img_name",
    )

    st.selectbox("Select Axis", ["z", "y", "x"], key="axis")
    max_slice_nr = mri_image.max_slice_nr(st.session_state.axis)
    st.slider("Select Slice", 0, max_slice_nr, max_slice_nr // 2, key="slice_nr")


with cols[1]:
    st.write("#### Brain Stats")
    st.metric(
        "Total Volume",
        f"{round(mri_image.volume_ml())} ml",
    )

    slice = mri_image.slice(st.session_state.axis, st.session_state.slice_nr)

    # st.metric(
    #     "Total Volume for Slice",
    #     f"{round(get_volume_in_ml_per_slice(slice, mri_image.voxel_size))} ml",
    # )

    st.metric(
        "Head Circumference",
        f"{round(mri_image.head_circumference(), 2)} cm",
    )

    # st.metric(
    #     "Total Volume for Slice (intensity)",
    #     f"{round(get_volume_in_ml_per_slice(slice, mri_image.voxel_size, st.session_state.intensity))} ml",
    # )


with cols[2]:
    st.write("#### GA Stats")
    st.metric(
        "Gestational Age",
        f"{round(mri_image.gestational_age(), 2)} weeks",
    )

    ga = predict_ga(mri_image)
    st.metric("Predicted GA (CNN)", f"{round(ga, 2)} weeks")

    ga_reg = predict_ga_reg(mri_image)
    st.metric("Predicted GA (Regression)", f"{round(ga_reg, 2)} weeks")
    st.metric("Predicted GA (Ensemble)", f"{round((ga + ga_reg) / 2, 2)} weeks")

with cols[3]:

    st.write(f"#### Brain MRI Slice {st.session_state.slice_nr}")

    st.pyplot(
        mri_image.slice(st.session_state.axis, st.session_state.slice_nr).plot(
            return_fig=True
        )
    )

    # with st.expander("Show animator"):
    #     st.plotly_chart(plot_animator(mri_image.slices_z), use_container_width=True)

with cols[4]:
    if st.checkbox("Show 3D Brain"):
        if f"3d_fig_{mri_image.filename}" not in st.session_state:
            st.session_state[f"3d_fig_{mri_image.filename}"] = plot_3d_brain(mri_image)
            st.plotly_chart(
                st.session_state[f"3d_fig_{mri_image.filename}"],
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                st.session_state[f"3d_fig_{mri_image.filename}"],
                use_container_width=True,
            )

cols = st.columns(2)

with cols[0]:
    volumes = [
        mri_image.volume_ml(intensity) for intensity in mri_image.intensity_values()
    ]

    st.plotly_chart(
        volume_histogram(volumes, mri_image.intensity_values()),
        use_container_width=True,
    )

with cols[1]:

    slice = mri_image.slice(st.session_state.axis, st.session_state.slice_nr)
    volumes = [
        slice.volume_in_ml(mri_image.voxel_size, intensity=intensity)
        for intensity in mri_image.intensity_values()
    ]
    st.plotly_chart(
        volume_histogram(volumes, mri_image.intensity_values()),
        use_container_width=True,
    )

import streamlit.components.v1 as components

st.divider()
st.write("### Linear Regression Plane for GA Prediction")
components.html(
    (PLOTS_PATH / "reg3d.html").read_text(),
    height=1000,
)


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
