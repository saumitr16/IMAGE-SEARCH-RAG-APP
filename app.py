import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a session with a longer timeout
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))



# Streamlit app title
st.title("Image Search API Client")

# Input for Ngrok link
ngrok_url = st.text_input("Enter your Ngrok URL (e.g., https://abcd-1234-5678.ngrok-free.app):")

# Check if Ngrok URL is provided
if not ngrok_url:
    st.warning("Please enter a valid Ngrok URL to proceed.")
else:
    st.success(f"Connected to FastAPI backend at: {ngrok_url}")

    # Sidebar for selecting the feature
    st.sidebar.title("Features")
    feature = st.sidebar.radio(
        "Choose a feature:",
        ("Upload Images", "Generate Captions", "Store Embeddings", "Search")
    )

    # Feature 1: Upload Images
    if feature == "Upload Images":
        st.header("Upload Images")
        query = st.text_input("Enter a text query (optional):")
        uploaded_file = st.file_uploader("Upload an image file (optional):", type=["jpg", "png", "jpeg"])

        if st.button("Upload"):
            if not query and not uploaded_file:
                st.error("Please provide either a text query or an image file.")
            else:
                files = {}
                data = {}
                if query:
                    data["query"] = query
                if uploaded_file:
                    files["file"] = uploaded_file.getvalue()

                response = requests.post(f"{ngrok_url}/upload/", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                    st.write(f"Number of images downloaded: {result['count']}")
                    if "generated_description" in result and result["generated_description"]:
                        st.write(f"Generated description: {result['generated_description']}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

    # Feature 2: Generate Captions
    elif feature == "Generate Captions":
        st.header("Generate Captions")
        if st.button("Generate Captions"):
            try:
                response = session.post(f"{ngrok_url}/generate_captions/", timeout=600, verify=False)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.SSLError as e:
                st.error(f"SSL Error: {e}")

    # Feature 3: Store Embeddings
    elif feature == "Store Embeddings":
        st.header("Store Embeddings")
        if st.button("Store Embeddings"):
            response = requests.post(f"{ngrok_url}/store_embeddings/")
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.write(f"Upserted files: {result['upserted_files']}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

    # Feature 4: Search
    elif feature == "Search":
        st.header("Search")
        mode = st.radio(
            "Search mode:",
            ("Image", "Text", "Both")
        )
        file = None
        text = None

        if mode in ["Image", "Both"]:
            file = st.file_uploader("Upload an image file:", type=["jpg", "png", "jpeg"])
        if mode in ["Text", "Both"]:
            text = st.text_input("Enter a text query:")

        if st.button("Search"):
            if mode == "Image" and not file:
                st.error("Please upload an image file.")
            elif mode == "Text" and not text:
                st.error("Please enter a text query.")
            elif mode == "Both" and (not file or not text):
                st.error("Please upload an image file and enter a text query.")
            else:
                files = {}
                data = {"mode": mode.lower()}
                if file:
                    files["file"] = file.getvalue()
                if text:
                    data["text"] = text

                response = requests.post(f"{ngrok_url}/search/", files=files, data=data)
                if response.status_code == 200:
                    results = response.json()["results"]
                    st.success("Search results:")
                    for result in results:
                        st.write(f"ID: {result['id']}")
                        st.write(f"Score: {result['score']}")
                        st.write(f"Caption: {result['caption']}")
                        st.write(f"Image Path: {result['image_path']}")
                        st.write("---")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")