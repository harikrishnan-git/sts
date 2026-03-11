import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from models.vit_encoder import ViTEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "weights/siamese_vit_fewshot9.pth"
SUPPORT_EMB_PATH = "support_embeddings.pt"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


@st.cache_resource
def load_model():

    model = ViTEncoder(embed_dim=256)

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource
def load_support():

    data = torch.load(SUPPORT_EMB_PATH)

    embeddings = data["embeddings"].to(DEVICE)
    labels = data["labels"]

    unique_labels = list(set(labels))

    prototypes = []

    for label in unique_labels:

        idx = [i for i,l in enumerate(labels) if l==label]

        proto = embeddings[idx].mean(0)

        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    return prototypes, unique_labels


model = load_model()

prototypes, class_names = load_support()


def predict(query_img):

    img_tensor = transform(query_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        query_emb = model(img_tensor)

        logits = -torch.cdist(query_emb, prototypes)

        pred = logits.argmax(dim=1)

    return class_names[pred.item()]


st.title("MRI Few-Shot Disease Classifier")

query = st.file_uploader("Upload MRI Image")

if query:

    img = Image.open(query).convert("RGB")

    st.image(img, width=300)

    if st.button("Predict"):

        pred = predict(img)

        st.success(f"Prediction: {pred}")