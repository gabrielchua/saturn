"""
app.py
"""
import asyncio
import pandas as pd
import streamlit as st

from utils import (
    set_page_configurations,
    set_session_state,
    set_custom_css,
    config_fail_validation,
    automatically_tag_async,
    )

set_page_configurations()
set_session_state()
set_custom_css()

st.title("SATURN 🪐")
st.caption("**S**imple **A**utomatic **T**agger yo**U** **R**eally **N**eeded")
st.subheader("Step 1: Upload your data")

file = st.file_uploader("Upload file",
                        type=["csv", "xlsx", "xls"],
                        label_visibility="hidden")

# Process uploaded file
if file is not None:
    # Read file
    if file.name.endswith("csv"):
        st.session_state["original_df"] = pd.read_csv(file)
    else:
        st.session_state["original_df"] = pd.read_excel(file)
    st.session_state["num_rows"] = st.session_state["original_df"].shape[0]


if st.session_state["original_df"] is not None:
    st.subheader("Step 2: Choose the column to tag")

    with st.expander("Preview Input Data"):
        st.write("Only the first 10 rows are shown")
        st.write(st.session_state["original_df"].head(10))

    str_columns = st.session_state["original_df"].select_dtypes(include="object").columns

    column_to_classify = st.selectbox("Choose the column you wish to tag", 
                                      str_columns,
                                      help="Only columns with text data can be tagged.")
        
    st.subheader("Step 3: Define the categories")

    number_categories = st.number_input("Number of categories",
                                        min_value=2,
                                        max_value=8,
                                        value=2,
                                        help="You can have a maximum of 10 categories. These categories should be mutually exclusive and collectively exhaustive.")

    configurations = pd.DataFrame({"One word label": [None]*number_categories, 
                                    "Description": [None]*number_categories, 
                                    "Example": [None]*number_categories})

    configurations.index = configurations.index+1

    st.session_state["tagging_configurations"] = st.data_editor(configurations, use_container_width=True)

    if st.button("Start Tagging"):
        status, error_message = config_fail_validation(st.session_state["tagging_configurations"])
        if status:
            st.warning(error_message, icon="🚨")
            st.stop()
        
        with st.spinner("Tagging in progress..."):
            st.session_state["tagged_df"] = asyncio.run(automatically_tag_async(st.session_state["original_df"],
                                                                                column_to_classify,
                                                                                st.session_state["tagging_configurations"]))
        
        st.session_state["tagged_df"] = st.session_state["tagged_df"].sort_values("original_row_number").drop(columns="original_row_number")

        st.subheader("Step 4: Review and Download")
        st.dataframe(st.session_state["tagged_df"], use_container_width=True)
        st.download_button("Download Tagged Data", 
                        data=st.session_state["tagged_df"].to_csv(index=False), 
                        file_name=f"{file.name[:-4]}_tagged.csv",
                        mime="text/csv")
