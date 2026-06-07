import streamlit as st
import cv2
import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import models, transforms
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
import pandas as pd
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Thermal Fault Diagnosis",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Thermal Fault Diagnosis System")
st.write("Upload a thermal image and diagnose the motor condition.")

# =====================================================
# LOAD RANDOM FOREST
# =====================================================

MODEL_PATH = "rf_model.pkl"

rf_model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD RESNET18
# =====================================================

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# MAINTENANCE RECOMMENDATIONS
# =====================================================

recommendations = {

    "Fan":
    "Inspect cooling fan condition. Check airflow, ventilation openings, fan blades and cooling efficiency.",

    "Rotor-0":
    "Inspect rotor bars, shaft alignment, rotor balance and vibration condition.",

    "Noload":
    "Motor operating normally. Continue periodic monitoring and preventive maintenance.",

    "A10":
    "Minor stator winding fault detected. Schedule preventive maintenance and thermal monitoring.",

    "A30":
    "Moderate stator winding degradation detected. Perform insulation testing and corrective maintenance.",

    "A50":
    "Severe stator winding short circuit detected. Immediate intervention recommended.",

    "A&C10":
    "Minor phase-to-phase degradation between phases A and C. Inspect winding insulation.",

    "A&C30":
    "Moderate phase-to-phase fault detected. Maintenance strongly recommended.",

    "A&B50":
    "Severe two-phase fault detected. Immediate corrective action required.",

    "A&C&B10":
    "Minor multi-phase anomaly detected. Increase monitoring frequency.",

    "A&C&B30":
    "Advanced multi-phase fault detected. Immediate diagnostic investigation required."
}

# =====================================================
# PREPROCESS + FEATURE EXTRACTION
# =====================================================

def extract_features(uploaded_image):

    img = np.array(uploaded_image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = img

    filtered = cv2.medianBlur(
        gray,
        3
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(filtered)

    normalized = enhanced.astype(
        np.float32
    ) / 255.0

    rows, cols = normalized.shape

    row_start = 39
    row_end = min(250, rows)

    col_start = 39
    col_end = min(300, cols)

    roi = normalized[
        row_start:row_end,
        col_start:col_end
    ]

    roi_uint8 = (
        roi * 255
    ).astype(np.uint8)

    roi_rgb = cv2.cvtColor(
        roi_uint8,
        cv2.COLOR_GRAY2RGB
    )

    pil_img = Image.fromarray(
        roi_rgb
    )

    img_tensor = transform(
        pil_img
    )

    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        features = feature_extractor(
            img_tensor
        )

    features = (
        features
        .squeeze()
        .cpu()
        .numpy()
    )

    return features.reshape(1, -1)

# =====================================================
# HEALTH INDEX
# =====================================================

def get_health_index(fault):

    health_table = {

        "Noload":100,

        "A10":80,

        "A&C10":75,

        "A&C&B10":70,

        "Fan":65,

        "Rotor-0":60,

        "A30":50,

        "A&C30":40,

        "A50":20,

        "A&B50":15,

        "A&C&B30":10
    }

    return health_table.get(
        fault,
        50
    )

# =====================================================
# RISK LEVEL
# =====================================================

def get_risk_level(fault):

    low_faults = [
        "Noload",
        "A10"
    ]

    medium_faults = [
        "Fan",
        "Rotor-0",
        "A&C10",
        "A&C&B10"
    ]

    if fault in low_faults:
        return "🟢 LOW"

    elif fault in medium_faults:
        return "🟡 MEDIUM"

    else:
        return "🔴 HIGH"
# =====================================================
# pdf
# =====================================================
def create_pdf_report(
    date,
    fault,
    confidence,
    health,
    risk,
    recommendation
):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    y = 800

    pdf.drawString(
        50,
        y,
        "THERMAL FAULT DIAGNOSIS REPORT"
    )

    y -= 40

    pdf.drawString(
        50,
        y,
        f"Date: {date}"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Fault: {fault}"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Confidence: {confidence:.2f}%"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Health Index: {health}/100"
    )

    y -= 25

    pdf.drawString(
        50,
        y,
        f"Risk Level: {risk}"
    )

    y -= 40

    pdf.drawString(
        50,
        y,
        "Recommendation:"
    )

    y -= 25

    lines = recommendation.split(".")

    for line in lines:

        if line.strip():

            pdf.drawString(
                70,
                y,
                line.strip()
            )

            y -= 20

    pdf.save()

    buffer.seek(0)

    return buffer

# =====================================================
# FILE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(

    "Upload Thermal Image",

    type=[
        "bmp",
        "jpg",
        "jpeg",
        "png"
    ]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Thermal Image",
            width=400
        )

    features = extract_features(image)

    features = pd.DataFrame(
        features,
        columns=rf_model.feature_names_in_
    )

    prediction = rf_model.predict(
        features
    )[0]

    probabilities = rf_model.predict_proba(
        features
    )[0]

    classes = rf_model.classes_

    best_confidence = (
        np.max(probabilities) * 100
    )

    current_date = datetime.now().strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    top_idx = np.argsort(
        probabilities
    )[::-1][:3]

    health = get_health_index(
        prediction
    )

    risk = get_risk_level(
        prediction
    )

    with col2:

        st.write(
            f"Inspection Date: {current_date}"
        )

        st.success(
            f"Predicted Fault: {prediction}"
        )

        st.metric(
            "Confidence",
            f"{best_confidence:.2f}%"
        )

        st.metric(
            "Health Index",
            f"{health:.1f}/100"
        )

        st.metric(
            "Risk Level",
            risk
        )

    st.subheader(
        "Top 3 Predictions"
    )

    for idx in top_idx:

        st.write(
            f"{classes[idx]} : "
            f"{probabilities[idx] * 100:.2f}%"
        )

    st.subheader(
        "Maintenance Recommendation"
    )

    st.info(
        recommendations.get(
            prediction,
            "No recommendation available."
        )
    )

    pdf_file = create_pdf_report(

        current_date,

        prediction,

        best_confidence,

        health,

        risk,

        recommendations.get(
            prediction,
            ""
        )
    )

    st.download_button(

        label="📄 Download PDF Report",

        data=pdf_file,

        file_name="Thermal_Report.pdf",

        mime="application/pdf"
    )