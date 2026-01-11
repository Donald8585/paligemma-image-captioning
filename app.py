import gradio as gr
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import PeftModel
import torch
from PIL import Image

print("🚀 Loading PaliGemma model...")

# Load base model
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load fine-tuned LoRA weights
model = PeftModel.from_pretrained(model, "Donald8585/paligemma-caption-finetuned")

# Load processor
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

print("✅ Model loaded successfully!")

def generate_caption(image):
    """Generate caption for uploaded image"""
    if image is None:
        return "Please upload an image!"
    
    # Process image
    inputs = processor(
        text="caption en",
        images=image,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    
    # Decode caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from output
    caption = caption.replace("caption en", "").strip()
    
    return caption

# Create Gradio interface
with gr.Blocks(title="PaliGemma Image Captioning") as demo:
    gr.Markdown("# 🎨 PaliGemma Image Captioning")
    gr.Markdown(
        """
        Fine-tuned PaliGemma-3B vision-language model for image captioning.
        Upload an image to generate a descriptive caption!
        
        **Model:** [Donald8585/paligemma-caption-finetuned](https://huggingface.co/Donald8585/paligemma-caption-finetuned)
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
