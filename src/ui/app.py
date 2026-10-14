"""
Streamlit frontend for the Smart Plant Health Monitor.

Talks to the FastAPI backend (src/api/main.py). Upload a leaf image to get a
disease classification from the ResNet50 model plus nutrient-need predictions
from the LSTM.

Run with:
    streamlit run src/ui/app.py

Set API_URL if the backend is not on http://localhost:8000.
"""

import os
import io

import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))

NEED_COLOURS = {
    "High Need": "#c0392b",
    "Medium Need": "#d4851a",
    "Low Need": "#2d7a3e",
}


st.set_page_config(
    page_title="Smart Plant Health Monitor",
    page_icon="leaf",
    layout="wide",
)


def api_available():
    """Return (ok, detail) for the backend health endpoint."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return True, r.json()
    except requests.RequestException as exc:
        return False, str(exc)


def analyze_image(file_bytes, filename):
    """POST the image to /analyze and return the parsed response."""
    files = {"file": (filename, file_bytes, "image/jpeg")}
    r = requests.post(f"{API_URL}/analyze", files=files, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def pretty_class_name(raw):
    """Turn 'Tomato___Late_blight' into 'Tomato / Late blight'."""
    if "___" in raw:
        crop, condition = raw.split("___", 1)
    else:
        crop, condition = raw, ""
    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return f"{crop} / {condition}" if condition else crop


def render_disease(disease):
    top = disease["top_prediction"]
    name = pretty_class_name(top["disease"])
    healthy = "healthy" in top["disease"].lower()

    st.subheader("Disease classification")
    if healthy:
        st.success(f"{name} ({top['percentage']} confidence)")
    else:
        st.warning(f"{name} ({top['percentage']} confidence)")

    rows = [
        {
            "Class": pretty_class_name(p["disease"]),
            "Confidence": p["confidence"],
        }
        for p in disease["top_3_predictions"]
    ]
    df = pd.DataFrame(rows).set_index("Class")
    st.caption(f"Top 3 of {disease['all_classes']} classes")
    st.bar_chart(df, y="Confidence", height=220)


def render_nutrition(nutrition):
    recs = nutrition["recommendations"]
    summary = nutrition["summary"]

    st.subheader("Nutrient needs")

    if summary["urgent_attention"]:
        st.error("Needs attention: " + ", ".join(summary["urgent_attention"]))
    elif summary["action_required"]:
        st.warning("Worth monitoring: " + ", ".join(summary["monitor"]))
    else:
        st.success("No urgent nutrient needs predicted.")

    df = pd.DataFrame(
        [
            {
                "Nutrient": r["nutrient"],
                "Need level": r["need_level"],
                "Confidence": f"{r['confidence'] * 100:.1f}%",
                "Low": r["probabilities"]["low"],
                "Medium": r["probabilities"]["medium"],
                "High": r["probabilities"]["high"],
            }
            for r in recs
        ]
    )

    def colour_need(val):
        return f"color: {NEED_COLOURS.get(val, 'inherit')}; font-weight: 600;"

    st.dataframe(
        df.style.map(colour_need, subset=["Need level"]).format(
            {"Low": "{:.2f}", "Medium": "{:.2f}", "High": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def main():
    st.title("Smart Plant Health Monitor")
    st.caption(
        "ResNet50 leaf disease classification (90.82% validation accuracy, "
        "38 classes) with LSTM nutrient-need forecasting."
    )

    ok, detail = api_available()

    with st.sidebar:
        st.header("Backend")
        st.code(API_URL, language=None)
        if ok:
            st.success("Connected")
        else:
            st.error("Not reachable")
            st.caption(detail)
            st.markdown(
                "Start it with:\n\n"
                "```bash\n"
                "python src/api/main.py\n"
                "```"
            )
        st.divider()
        st.markdown(
            "**Models**\n\n"
            "- ResNet50, 38 classes, 90.82% val accuracy\n"
            "- YOLOv5, 5 classes, 86.6% mAP@0.5\n"
            "- LSTM, 30-day sequences, 69.4% val accuracy"
        )

    uploaded = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        help="A single leaf, reasonably well lit, fills most of the frame.",
    )

    if uploaded is None:
        st.info("Upload a leaf image to run the analysis.")
        return

    file_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with right:
        if not ok:
            st.error("Cannot analyse while the backend is unreachable.")
            return

        with st.spinner("Running models..."):
            try:
                result = analyze_image(file_bytes, uploaded.name)
            except requests.HTTPError as exc:
                st.error(f"Backend returned an error: {exc.response.status_code}")
                st.caption(exc.response.text[:500])
                return
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        render_disease(result["disease_analysis"])
        st.divider()
        # /analyze returns the nutrition payload unwrapped; /nutrition nests it
        # under "predictions". Accept either shape.
        nutrition = result["nutritional_analysis"]
        render_nutrition(nutrition.get("predictions", nutrition))

    st.divider()
    st.caption(
        "Nutrient predictions in this demo run on synthetic sensor sequences. "
        "Wire real sensor readings into /nutrition for production use."
    )


if __name__ == "__main__":
    main()
