import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://lh3.googleusercontent.com/drive-viewer/AKGpihYnuF2DxoLEUWoXfNqUGdawFKUVwNd7MUifuSHqGBIxWuTBV6oXzVM-FAK_T92_CPUedlWSiWLMuW0GudpZTnnscQ5DEA=s1600');
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 50px 20px;
                background-size: 13em 13em;
            }
            [data-testid="stSidebarNav"]::before {
                content: "OptiPromo";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
                align-items: center;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
