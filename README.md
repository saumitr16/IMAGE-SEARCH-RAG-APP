# IMAGE-SEARCH-RAG-APP


ImageSearch_RAG_App is an end-to-end system that demonstrates an image search and caption generation workflow. The project consists of:

- **A FastAPI-powered backend (`api.py`)** that processes image uploads, generates captions, stores embeddings, and supports search functionality.
- **A Streamlit-powered frontend (`app.py`)** that provides a user-friendly interface to interact with the API.
- **A demo notebook (`Mindlfix_Assignment.ipynb`)** that walks through the backend functionalities, offering a detailed explanation of each processing step.

> **Note:** Two sample folders are also attached to store the downloaded images and captions in the backend.

---

## Project Structure

### 1. `api.py`
- Implements the FastAPI backend with endpoints to:
  - Upload images and/or text queries (e.g., `POST` to `/upload/`).
  - Generate captions based on the uploaded image.
  - Compute and store embeddings for images.
  - Enable search within stored embeddings.
- Handles request processing and returns responses in JSON format.
- All the API call commands can be read through `API_Details.txt` file.

### 2. `app.py`
- A Streamlit-based client that provides an interactive UI.
- Run the app by 'streamlit run app.py'
- **Key Features:**
  - Ngrok integration: Input field to connect to your FastAPI backend via a public URL.
  - Sidebar option to select between various features:
    - Upload Images
    - Generate Captions
    - Store Embeddings
    - Search
  - Sends HTTP requests to the backend API and displays the results (e.g., success messages, generated image captions, image counts).

### 3. `Mindlfix_Assignment.ipynb`
- A demonstration notebook that serves as a demo version of the backend.
- **Includes:**
  - Installation commands for dependencies like `clip`, `transformers`, `torch`, `torchvision`, `pillow`, and `pinecone-client`.
  - Step-by-step walkthrough of backend operations such as image uploading, caption generation, and embedding computation.
  - Detailed explanations of how the backend processes requests and interacts with the frontend.

---

## Installation

### 1. Clone the Repository:
```bash
$ git clone <repository_url>
```

### 2. Navigate to the Project Folder:
```bash
$ cd Mindflix_RAG_Project
```

### 3. Install the required packages:

**For the backend and frontend:**
```bash
$ pip install streamlit fastapi uvicorn requests pillow
```

**For the demo notebook (`Mindlfix_Assignment.ipynb`):**
```bash
$ pip install clip
$ pip install transformers torch torchvision pillow
$ pip install pinecone-client transformers torch torchvision pillow
```

---

## Running the Application

### Step 1: Start the Backend API
Run the `api.py` file and you will see the URL of the localhost server.

Run the FastAPI backend using the `api.py` file via Uvicorn:
```bash
$ uvicorn api:app --reload
```

If you are using Ngrok to expose your local server, launch Ngrok with:
```bash
$ ngrok http 8000
```

Note the provided public URL.

### Step 2: Launch the Frontend Client
Run the Streamlit-based frontend using the `app.py` file:
```bash
$ streamlit run app.py
```

When the Streamlit UI loads, enter the Ngrok URL to connect to the backend API.

---

## Application Flow

### Step 1: Inserting the Ngrok URL
This field was necessary because Ngrok changes IP address frequently in the free tier. Just input the Ngrok URL received after running `api.py`.

### Step 2: Uploading Images
Two options are available:
- Upload Image
- Upload Text

You can do both and click on the **Upload** button to start uploading in the backend.

### Step 3: Generating Captions
Just click on the **Generate Captions** button and wait for a suitable time (nearly 5 minutes for 50 images).

### Step 4: Inserting Embeddings in the Pinecone DB
Just click on the **Store Embeddings** button.

### Step 5: Searching
You have two options:
- Input image
- Input text

You can also use both. The top 5 image results (image paths) with high-quality metadata will be shown.

---

## Project Flow & Functioning

1. **Backend Initialization (`api.py`):**
   - FastAPI starts up and exposes endpoints to receive and process various requests (upload, caption, embedding, search).

2. **User Interaction (`app.py`):**
   - The Streamlit client prompts the user to supply the Ngrok URL.
   - Users select a feature from the sidebar (e.g., Upload Images) and either provide a text query or upload an image.
   - The client sends relevant HTTP requests to the backend.

3. **Processing & Response (`api.py`):**
   - The backend processes the input (image file/text query), executes the corresponding functionality (such as generating captions or computing embeddings), and then returns the results as JSON.

4. **Results Display (`app.py`):**
   - The Streamlit client displays the processed results to the user including success messages, caption text, image counts, or search results.

5. **Demo Notebook (`Mindlfix_Assignment.ipynb`):**
   - Provides a comprehensive demo, including detailed steps of the backend function.
   - Illustrates package installations, request processing, data handling, and backend logic.
   - Serves as a tutorial for understanding the internal workings of the project.

---

## Conclusion

This project offers a modular and comprehensive approach to image search and caption generation:
- The separation of backend (FastAPI) and frontend (Streamlit) ensures cleaner architecture and easier testing.
- The demo notebook (`Mindlfix_Assignment.ipynb`) is an invaluable resource for learning and troubleshooting the backend processes.
- Enjoy exploring, extending, and deploying the Mindflix system for your own unique use cases!

For any questions or contributions, please consult the project documentation or reach out on our repository’s issue tracker.
