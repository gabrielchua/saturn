"""
app.py
"""
import asyncio
import pandas as pd
import streamlit as st

from utils import (
    config_fail_validation,
    render_download_page,
    set_custom_css,
    set_page_configurations,
    set_session_state,
    start_async_tag_job
    )

set_page_configurations()
set_session_state()
set_custom_css()

st.title("SATURN 🪐")
st.caption("**S**imple **A**utomatic **T**agger yo**U** **R**eally **N**eeded")
st.subheader("Step 1: Upload your data")

uploaded_file = st.file_uploader("Upload file",
                                 type=["csv", "xlsx", "xls"],
                                 label_visibility="hidden")

# Process uploaded file
if uploaded_file is not None:

    # Save the file name
    st.session_state.uploaded_file_name = uploaded_file.name

    # Read file
    if st.session_state.uploaded_file_name.endswith("csv"):
        st.session_state["original_df"] = pd.read_csv(uploaded_file)
    else:
        st.session_state["original_df"] = pd.read_excel(uploaded_file)
    st.session_state["num_rows"] = st.session_state["original_df"].shape[0]

# This will load if there is an uploaded file
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

    tab_1, tab_2 = st.tabs(["Manual", "Upload Mapping Table"])

    number_categories = tab_1.number_input("Number of categories",
                                        min_value=2,
                                        max_value=8,
                                        value=2,
                                        help="You can have a maximum of 10 categories. These categories should be mutually exclusive and collectively exhaustive.")
    
    # Allow users to upload their own mapping table
    uploaded_mapping_table = tab_2.file_uploader("Upload mapping table",
                                     type=["csv", "xlsx", "xls"])
    
    st.divider()

    # This will generate a blank dataframe
    configurations = pd.DataFrame({"One word label": [None]*number_categories,
                                   "Description": [None]*number_categories,
                                   "Example": [None]*number_categories})

    configurations.index = configurations.index+1

    st.session_state["tagging_configurations"] = tab_1.data_editor(configurations, use_container_width=True)

    # This is the dataframe editor
    if uploaded_mapping_table is not None:
        st.session_state["tagging_configurations"] = pd.read_csv(uploaded_mapping_table)
        tab_2.dataframe(st.session_state["tagging_configurations"], use_container_width=True)

    if st.button("Start Tagging"):
        status, error_message = config_fail_validation(st.session_state["tagging_configurations"])
        if status:
            st.warning(error_message, icon="🚨")
            st.stop()
        
        with st.spinner("Tagging in progress..."):
            st.session_state["tagged_df"] = asyncio.run(start_async_tag_job(st.session_state["original_df"],
                                                                            column_to_classify,
                                                                            st.session_state["tagging_configurations"]))
        
        st.session_state["tagged_df"] = st.session_state["tagged_df"].sort_values("original_row_number").drop(columns="original_row_number")

        st.subheader("Step 4: Review and Download")
        st.dataframe(st.session_state["tagged_df"], use_container_width=True)
        render_download_page()
