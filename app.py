import gradio as gr
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import PeftModel
import torch
from PIL import Image
import os
import spaces

print("🚀 Loading PaliGemma model...")

# Get HuggingFace token from environment
hf_token = os.getenv("HF_TOKEN")

# Load base model on CPU first
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.bfloat16,
    token=hf_token
)

# Load fine-tuned LoRA weights
model = PeftModel.from_pretrained(
    model, 
    "Donald8585/paligemma-caption-finetuned",
    token=hf_token
)

# Move model to CUDA (Zero GPU manages this!)
model.to('cuda')

# Load processor
processor = PaliGemmaProcessor.from_pretrained(
    "google/paligemma-3b-pt-224",
    token=hf_token
)

print("✅ Model loaded successfully!")

@spaces.GPU(duration=120)
def generate_caption(image):
    """Generate caption for uploaded image"""
    if image is None:
        return "Please upload an image!"
    
    try:
        # Convert image to RGB (handles GIFs, grayscale, RGBA)
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Process image
        inputs = processor(
            text="caption en",
            images=image,
            return_tensors="pt"
        ).to('cuda')
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        # Decode caption
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        caption = caption.replace("caption en", "").strip()
        
        return caption
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="PaliGemma Image Captioning") as demo:
    gr.Markdown("# 🎨 PaliGemma Image Captioning")
    gr.Markdown(
        """
        Fine-tuned PaliGemma-3B vision-language model for image captioning.
        Upload an image to generate a descriptive caption!
        
        **Model:** [Donald8585/paligemma-caption-finetuned](https://huggingface.co/Donald8585/paligemma-caption-finetuned)
        
        ✨ **Supports:** PNG, JPG, JPEG, GIF, WEBP
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Generate Caption", variant="primary")
        
        with gr.Column():
            caption_output = gr.Textbox(label="Generated Caption", lines=3)
    
    # Examples
    gr.Examples(
        examples=[
            ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"],
            ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"],
        ],
        inputs=image_input,
    )
    
    submit_btn.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=caption_output
    )
    
    gr.Markdown("---")
    gr.Markdown(
        """
        ### About
        - **Developer:** Alfred So | [LinkedIn](https://www.linkedin.com/in/alfred-so/) | [GitHub](https://github.com/Donald8585/)
        - **Tech:** PaliGemma-3B + LoRA fine-tuning
        - **Training:** Custom dataset, Google Colab T4 GPU
        """
    )

if __name__ == "__main__":
    demo.launch()
