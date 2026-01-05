import streamlit as st
import requests

API_URL = "http://localhost:8000/api/qa/"

st.set_page_config(page_title="Vision RAG")

st.title("Vision RAG")
st.write("Upload a document once, then ask grounded questions.")
st.divider()


if "doc_id" not in st.session_state:
    st.session_state.doc_id = None


if st.session_state.doc_id:
    if st.button("Reset document"):
        st.session_state.doc_id = None
        st.experimental_rerun()


document_file = None
if st.session_state.doc_id is None:
    document_file = st.file_uploader(
        label="Upload a PDF (required once)",
        type="pdf"
    )
else:
    st.success("Document loaded. You can ask questions below.")

st.divider()


query_text = st.text_input(
    label="Ask a question",
    placeholder="e.g., What does Figure 2 show?"
)


if st.button("Ask"):
    if not query_text:
        st.warning("Please enter a question.")
        st.stop()

    if st.session_state.doc_id is None and document_file is None:
        st.error("Please upload a document before asking questions.")
        st.stop()

    with st.spinner("Querying document..."):
        data = {"question": query_text}
        files = {}

        if document_file is not None:
            files["file"] = (
                document_file.name,
                document_file.getvalue(),
                "application/pdf"
            )

        if st.session_state.doc_id is not None:
            data["doc_id"] = st.session_state.doc_id

        try:
            resp = requests.post(API_URL, data=data, files=files, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

        result = resp.json()

        if result.get("doc_id"):
            st.session_state.doc_id = result["doc_id"]


    st.subheader("Answer")
    st.write(result.get("answer", "No answer returned."))


    candidates = result.get("candidates", [])

    if candidates:
        st.divider()
        st.subheader("Retrieved Evidence")

        for i, c in enumerate(candidates, start=1):
            meta = c["metadata"]
            st.markdown(f"**Result {i}**")
            st.markdown(f"- Type: `{meta.get('type')}`")
            st.markdown(f"- Score: `{c['score']:.3f}`")

            if meta.get("caption"):
                st.markdown("**Caption:**")
                st.code(meta["caption"][:800])
            else:
                st.markdown("**Content:**")
                st.code(meta["content"][:800])
    else:
        st.info("No supporting document evidence returned.")