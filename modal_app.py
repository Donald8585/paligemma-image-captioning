import modal
from pathlib import Path

# Create Modal app
app = modal.App("paligemma-fashion-caption")

# Create persistent volume for model cache
model_volume = modal.Volume.from_name("paligemma-models", create_if_missing=True)
MODEL_DIR = "/models"

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "torch>=2.0.0",
        "Pillow>=10.0.0",
        "accelerate>=0.27.0",
        "fastapi",
        "python-multipart",
    )
)

@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
class PaliGemmaModel:
    @modal.enter()
    def load_model(self):
        """Load model into GPU memory on container start"""
        import os
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        from peft import PeftModel
        import torch
        
        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = MODEL_DIR
        
        print("🚀 Loading PaliGemma model...")
        
        # Load base model (cached in volume)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-pt-224",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            cache_dir=cache_dir
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            "Donald8585/paligemma-caption-finetuned",
            token=hf_token,
            cache_dir=cache_dir
        )
        
        self.model.to('cuda')
        self.model.eval()
        
        # Load processor
        self.processor = PaliGemmaProcessor.from_pretrained(
            "google/paligemma-3b-pt-224",
            token=hf_token,
            cache_dir=cache_dir
        )
        
        print("✅ Model loaded successfully!")
    
    @modal.method()
    def generate_caption(self, image_bytes: bytes) -> str:
        """Generate caption from image bytes"""
        import torch
        from PIL import Image
        from io import BytesIO
        
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Process image - UPDATED PROMPT!
        inputs = self.processor(
            text="caption",  # ✅ Changed from "caption en"
            images=image,
            return_tensors="pt"
        ).to('cuda')
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)  # Increased tokens
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        caption = caption.replace("caption", "").strip()
        
        return caption

# FastAPI web endpoint with CORS
@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import base64
    
    web_app = FastAPI()
    
    # Enable CORS for cross-origin requests
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @web_app.get("/")
    async def root():
        return {"message": "PaliGemma Fashion Caption API", "model": "paligemma-caption-finetuned"}
    
    @web_app.post("/generate")
    async def generate_caption_endpoint(file: UploadFile = File(...)):
        """Accept image upload and return caption"""
        try:
            image_bytes = await file.read()
            model = PaliGemmaModel()
            caption = model.generate_caption.remote(image_bytes)
            return JSONResponse({"caption": caption})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    @web_app.post("/generate-base64")
    async def generate_caption_base64(data: dict):
        """Accept base64 image and return caption"""
        try:
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            model = PaliGemmaModel()
            caption = model.generate_caption.remote(image_bytes)
            return JSONResponse({"caption": caption})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return web_app
