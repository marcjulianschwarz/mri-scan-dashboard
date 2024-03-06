import streamlit as st

st.set_page_config(
    page_title="Brain MRI - Dashboard - 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.write("# 🧠 Brain MRI - Dashboard")
st.write(
    "Welcome to the brain MRI dashboard. Select a page from the sidebar or use the links below to navigate."
)
st.write("### Pages")
st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/00_Single Brain.py", label="Single Brain", icon="🧠")
st.page_link("pages/01_All Brains.py", label="All Brains", icon="📊")


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
