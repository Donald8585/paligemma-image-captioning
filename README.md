# 🎨 PaliGemma Fine-tuning for Image Captioning

Fine-tuned Google's PaliGemma-3B vision-language model using LoRA (Low-Rank Adaptation) for image captioning tasks.

## 🚀 Project Overview

This project demonstrates fine-tuning of Google's state-of-the-art PaliGemma vision-language model on a custom image captioning dataset. Using parameter-efficient fine-tuning (PEFT) with LoRA, the model was trained to generate accurate captions for images.

## 🛠️ Tech Stack

- **Model**: PaliGemma-3B (Google)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch, Hugging Face Transformers
- **Libraries**: PEFT, Accelerate, BitsAndBytes
- **Training**: Google Colab with T4 GPU

## 📊 Model Details

- **Base Model**: `google/paligemma-3b-pt-224`
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 8
  - Target modules: q_proj, v_proj
  - Dropout: 0.05
- **Training**:
  - Epochs: 1
  - Batch size: 2
  - Learning rate: 1e-4
  - Precision: BF16

## 🎯 Results

Successfully fine-tuned the model to generate descriptive captions:

| Input Image | Generated Caption |
|------------|------------------|
| Car | "automobile model parked on a street" |
| Tiger | "animal lying down in the grass" |

## 📁 Repository Contents

- `paligemma_finetuning.ipynb` - Complete training notebook
- `paligemma-finetuned-lora.zip` - Fine-tuned model weights (LoRA adapters)

## 🔧 Usage

```python
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import PeftModel
import torch

# Load base model
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "./paligemma-finetuned-lora")
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

# Generate caption
from PIL import Image
image = Image.open("your_image.jpg")
inputs = processor(text="caption en", images=image, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

## 🎓 Learning Outcomes

- Implemented parameter-efficient fine-tuning using LoRA
- Worked with vision-language models (VLMs)
- Handled multi-modal data processing (images + text)
- Optimized training for GPU memory constraints
- Applied modern ML engineering practices

## 📚 References

- [PaliGemma Model Card](https://huggingface.co/google/paligemma-3b-pt-224)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 👨‍💻 Author

**Alfred So** | [LinkedIn](https://www.linkedin.com/in/alfred-so/) | [GitHub](https://github.com/Donald8585/)

---

*Part of my AI/ML portfolio - showcasing hands-on experience with vision-language models and modern fine-tuning techniques.*
