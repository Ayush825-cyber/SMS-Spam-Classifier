import streamlit as st
import joblib
import text_utils  # noqa: F401  (needed to load the saved model)

# Load trained model
model = joblib.load("spam_model.pkl")

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Header
st.title("📩 SMS Spam Detector")
st.caption("Machine Learning powered message classification")

st.divider()



# Message input
message = st.text_area(
    "Enter your SMS message",
    key="message",
    height=130,
    placeholder="Type or paste your message here..."
)

# Prediction
if st.button("🔍 Analyze Message", use_container_width=True):

    if message.strip():

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        # Get probability for the predicted class
        classes = model.named_steps["classifier"].classes_
        confidence = probabilities[list(classes).index(prediction)]

        st.divider()
        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM")

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(float(confidence))

        st.caption(
            "Confidence represents the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

    else:
        st.warning("Please enter a message first.")

st.divider()

st.caption("Built with Python, Scikit-learn and Streamlit")