import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import requests

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("weights")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def add_caption_to_image(image, caption):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img.width - text_width) // 2
    y = img.height - text_height - 20

    draw.rectangle(
        [(x-5, y-5), (x + text_width + 5, y + text_height + 5)],
        fill="black"
    )
    draw.text((x, y), caption, font=font, fill="white")
    
    return img


def main():
    st.title("Meme captions generator")

    processor, model, device = load_model()

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Enter image url")
    
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
    elif image_url:
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        except:
            st.error("Unable to laod image")
    
    if image:
        st.image(image, caption="Source image", use_container_width=True)
        
        if st.button("Generate"):
            with st.spinner("Generating..."):
                try:
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    pixel_values = inputs.pixel_values

                    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
                    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    result_image = add_caption_to_image(image, generated_caption)

                    st.subheader("Generated caption:")
                    st.write(generated_caption)
                    
                    st.subheader("Result:")
                    st.image(result_image, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Errorr: {str(e)}")

if __name__ == "__main__":
    main()