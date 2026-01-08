import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Step 1: Configure Streamlit page
st.set_page_config(page_title="CPU Image Classifier", layout="centered")
st.title("🖼️ CPU-based Image Classification with ResNet18")
st.write("Upload an image to get the top-5 predictions using a pretrained ResNet18 model.")

# Step 3: CPU configuration
device = torch.device("cpu")

# Step 4: Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model.to(device)

# Step 5: Define preprocessing transforms
preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# Step 6: Image upload UI
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Step 7: Preprocess image and convert to tensor
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Step 7: Inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Step 8: Softmax and Top-5 predictions
    probs = F.softmax(outputs, dim=1)
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    # Get human-readable class names
    class_names = models.ResNet18_Weights.DEFAULT.meta["categories"]
    
    # Prepare dataframe for display
    top5_df = pd.DataFrame({
        "Class": [class_names[idx] for idx in top5_catid[0]],
        "Probability": top5_prob[0].cpu().numpy()
    })
    
    st.subheader("Top-5 Predictions")
    st.table(top5_df)
    
    # Step 9: Visualize probabilities
    st.subheader("Prediction Probabilities")
    st.bar_chart(top5_df.set_index("Class"))