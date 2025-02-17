from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from dotenv import load_dotenv
import os

# Loading environment variables
load_dotenv()

# Access environment variables
pinecone_key = os.getenv("pinecone_key")
pexels_key= os.getenv("pexels_key")
ngork= os.getenv("ngork")



# Initializing FastAPI app
app = FastAPI()

# Initializing models and Pinecone
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
pc = Pinecone(api_key=pinecone_key)  
index_name = "image-search"
index = pc.Index(index_name)

# Helper functions
def pexels_image_search(query, num_images=50):
    """Fetches image URLs from Pexels based on a text query."""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": pexels_key} 
    params = {"query": query, "per_page": num_images}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        results = response.json()
        return [photo["src"]["original"] for photo in results.get("photos", [])]
    else:
        print(f"❌ Pexels API error: {response.status_code}")
        return []

def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)


def download_images(image_urls, save_dir="downloaded_images"):
    """Downloads images from a list of URLs and saves them locally."""
    os.makedirs(save_dir, exist_ok=True)
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            img.save(f"{save_dir}/image_{i+1}.jpg")
            print(f"✅ Downloaded: {url}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    return image_embedding.squeeze().cpu().numpy()

def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding.squeeze().cpu().numpy()

# FastAPI endpoints
@app.post("/upload/")
async def upload_images(query: str = Form(None), file: UploadFile = File(None)):
    image_urls = []
    if query:
        # If a text query is provided, search Pexels for images
        image_urls += pexels_image_search(query)
    if file:
        # If an image file is provided, generate a description and search Pexels
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Generate a description of the uploaded image
        description = generate_image_description(image_path)
        print(f"Generated description: {description}")
        
        # Search Pexels using the generated description
        image_urls += pexels_image_search(description)
    
    # Download the images
    download_images(image_urls)
    
    return JSONResponse(
        content={
            "message": "Images downloaded successfully",
            "count": len(image_urls),
            "generated_description": description if file else None  # Include the generated description if a file was uploaded
        }
    )

@app.post("/generate_captions/")
async def generate_captions():
    image_folder = "downloaded_images"
    output_folder = "downloaded_captions"
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            inputs = blip_processor(images=image, return_tensors="pt")
            output = blip_model.generate(**inputs)
            caption = blip_processor.decode(output[0], skip_special_tokens=True)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, "w") as f:
                f.write(caption)
    return JSONResponse(content={"message": "Captions generated successfully"})

@app.post("/store_embeddings/")
async def store_embeddings():
    image_folder = "downloaded_images"
    caption_folder = "downloaded_captions"
    upserted_files = []  # Track successfully upserted files

    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, filename)
            caption_path = os.path.join(caption_folder, os.path.splitext(filename)[0] + ".txt")

            # Read caption
            try:
                with open(caption_path, "r") as f:
                    caption = f.read().strip()
            except Exception as e:
                print(f"Failed to read caption for {filename}: {e}")
                continue

            # Generate embeddings
            try:
                image_embedding = get_image_embedding(image_path)
                text_embedding = get_text_embedding(caption)
            except Exception as e:
                print(f"Failed to generate embeddings for {filename}: {e}")
                continue

            # Combine embeddings
            combined_embedding = np.concatenate((image_embedding, text_embedding))

            # Prepare metadata
            metadata = {
                "caption": caption,
                "image_path": image_path  # Store the path or URL of the image
            }

            # Check if filename already exists in Pinecone
            try:
                existing_results = index.query(
                    vector=combined_embedding.tolist(),
                    top_k=1,
                    include_metadata=True,
                    filter={"id": filename}  # Filter by filename
                )

                if existing_results["matches"]:
                    print(f"Updating existing entry for {filename} in Pinecone...")
                    index.delete(ids=[filename])  # Remove old entry if it exists

                # Upsert to Pinecone
                index.upsert([(filename, combined_embedding.tolist(), metadata)], timeout=10)
                upserted_files.append(filename)
                print(f"Upserted {filename} to Pinecone")
            except Exception as e:
                print(f"Failed to upsert {filename}: {e}")

    return JSONResponse(
        content={
            "message": "Embeddings stored in Pinecone successfully",
            "upserted_files": upserted_files,
        }
    )

@app.post("/search/")
async def search(mode: str = Form(...), file: UploadFile = File(None), text: str = Form(None)):
    try:
        if mode == "image" and file:
            # Processing image
            image_path = f"/tmp/{file.filename}"
            with open(image_path, "wb") as buffer:
                buffer.write(await file.read())
            image_embedding = get_image_embedding(image_path)
            query_vector = np.pad(image_embedding, (0, 512))
        elif mode == "text" and text:
            # Processing text
            text_embedding = get_text_embedding(text)
            query_vector = np.pad(text_embedding, (512, 0))
        elif mode == "both" and file and text:
            # Processing both image and text
            image_path = f"/tmp/{file.filename}"
            with open(image_path, "wb") as buffer:
                buffer.write(await file.read())
            image_embedding = get_image_embedding(image_path)
            text_embedding = get_text_embedding(text)
            query_vector = np.concatenate((image_embedding, text_embedding))
        else:
            raise HTTPException(status_code=400, detail="Invalid input for the selected mode")

        # Query Pinecone
        results = index.query(vector=query_vector.tolist(), top_k=5, include_metadata=True)

        # Converting ScoredVector objects to a JSON-serializable format
        serializable_results = []
        for match in results["matches"]:
            serializable_results.append({
                "id": match.id,
                "score": match.score,
                "caption": match.metadata.get("caption", ""),  # Extracting caption from metadata
                "image_path": match.metadata.get("image_path", ""),  # Extracting image path from metadata
                "metadata": match.metadata  # Including all metadata if needed
            })

        return JSONResponse(content={"results": serializable_results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Search API!"}

# # Configure ngrok authtoken
ngrok.set_auth_token(ngork)

# # Start the FastAPI app with ngrok
ngrok_tunnel = ngrok.connect(8001)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8001)