import streamlit as st

st.set_page_config(
    page_title="UI",
    page_icon="",
)

st.title("UI")
st.write(
    "A simple interface to upload a document,image, and ask questions"
)
st.divider() 


query_text = st.text_input(
    label="Ask a question",
    placeholder="e.g., What is shown in the first chart?"
)

document_file = st.file_uploader(
    label="Upload a context document (PDF)",
    type="pdf"
)


image_file = st.file_uploader(
    label="Upload an image",
    type=['png', 'jpg', 'jpeg']
)

if st.button("Process Inputs"):
    st.divider()

    if query_text:
        st.subheader("Text Query Received:")
        st.write(query_text)
    else:
        st.warning("No text query was provided.")

    if document_file is not None:
        st.subheader("Document File Received:")
        st.write(f"**File Name:** `{document_file.name}`")
    else:
        st.warning("No document file was uploaded.")

    if image_file is not None:
        st.subheader("Image File Received:")
        st.write(f"**File Name:** `{image_file.name}`")
        st.image(image_file, caption="Uploaded Image.")
