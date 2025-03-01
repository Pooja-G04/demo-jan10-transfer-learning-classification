import torch
import torchvision.models as models
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from data_preprocessing import prepare_data  # Import your data preprocessing script
from torchvision.models import wide_resnet50_2  # _2 indicates a width multiplier of 2



def load_model(weights_path="model_weights.pth", num_classes=10):
    """
    Loads the trained model with weights.

    Args:
        weights_path (str): Path to the .pth file containing trained weights.
        num_classes (int): Number of output classes (should match training).

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Load a pre-trained model (ResNet18 as an example)
    model = wide_resnet50_2(pretrained=True) 
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust final layer

    # Load the trained weights
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))

    # Set model to evaluation mode
    model.eval()
    
    return model

# Prepare data and get class names dynamically
_, _, class_names, device = prepare_data()

# Load model and set to evaluation mode
model = load_model(weights_path = "model_wideresnet50_run0.pth")  # Load your trained model here
model.to(device)
model.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_car_label(image):
    """
    Predicts the car model for a given image.

    Args:
        image (PIL.Image): Input image.

    Returns:
        str: Predicted car model name.
    """
    image = transform(image).unsqueeze(0).to(device)  # Preprocess and add batch dim
    with torch.no_grad():
        output = model(image)
        _, pred_index = torch.max(output, 1)
    return class_names[pred_index.item()]

# Streamlit UI
st.title("Car Model Classifier 🚗")

uploaded_file = st.file_uploader("Upload an image of a car...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        predicted_label = predict_car_label(image)
        st.success(f"Predicted Car Model: **{predicted_label}**")
