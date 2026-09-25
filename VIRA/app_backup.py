import streamlit as st
import json

from backend.audio_processor import get_audio_info
from backend.deepfake_detector import detect_deepfake
from backend.speaker_verifier import verify_speaker
from backend.risk_engine import calculate_risk


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="VIRA",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #F8FAFC;
}

.stMetric {
    background-color: #111827;
    border: 1px solid #1E293B;
    padding: 15px;
    border-radius: 10px;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
}

</style>
""", unsafe_allow_html=True)

st.title("VIRA")
st.caption("Voice Impersonation Risk Assessment")

st.write(
    "AI-powered voice impersonation and voice-cloning risk assessment"
)


# -----------------------------------
# Audio Upload
# -----------------------------------

st.divider()

st.subheader("Call Audio Input")

audio_file = st.file_uploader(
    "Upload a call recording",
    type=["wav", "mp3", "ogg", "m4a"]
)


# -----------------------------------
# Analysis
# -----------------------------------

if audio_file is not None:

    audio_info = get_audio_info(audio_file)

    st.success(audio_info["status"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**File:**", audio_info["file_name"])

    with col2:
        st.write(
            "**Size:**",
            f"{audio_info['file_size_bytes'] / 1024:.1f} KB"
        )

    if st.button("Analyze Call", type="primary"):

        with st.spinner("Analyzing call..."):

            # Deepfake detection
            deepfake_result = detect_deepfake(audio_file)

            # Speaker verification
            speaker_result = verify_speaker(audio_file)

            # Temporary context risk for Day-1 MVP
            context_risk = 0.78

            # Risk engine
            risk_result = calculate_risk(
                deepfake_result["deepfake_probability"],
                speaker_result["speaker_match_probability"],
                context_risk
            )

        st.success("Analysis completed")


        # -----------------------------------
        # Risk Scores
        # -----------------------------------

        st.divider()

        st.subheader("Risk Assessment")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Deepfake Probability",
                f"{deepfake_result['deepfake_probability'] * 100:.0f}%"
            )

        with col2:
            st.metric(
                "Speaker Match",
                f"{speaker_result['speaker_match_probability'] * 100:.0f}%"
            )

        with col3:
            st.metric(
                "Context Risk",
                f"{context_risk * 100:.0f}%"
            )

        with col4:
            st.metric(
                "Final Risk",
                f"{risk_result['final_risk_score'] * 100:.0f}%"
            )


        # -----------------------------------
        # Recommended Action
        # -----------------------------------

        st.subheader("Recommended Action")

        final_score = risk_result["final_risk_score"] * 100

        if final_score >= 80:
            st.error(risk_result["recommended_action"])

        elif final_score >= 50:
            st.warning(risk_result["recommended_action"])

        else:
            st.success(risk_result["recommended_action"])


        # -----------------------------------
        # Detection Results
        # -----------------------------------

        st.divider()

        st.subheader("Detection Results")

        st.write(
            f"**Deepfake Analysis:** "
            f"{deepfake_result['reason_code']}"
        )

        st.write(
            f"**Speaker Verification:** "
            f"{speaker_result['reason_code']}"
        )


        # -----------------------------------
        # Chunk-wise sample analysis
        # -----------------------------------

        st.divider()

        st.subheader("Chunk-wise Analysis")

        try:

            with open("sample_chunk_scores.json", "r") as file:
                data = json.load(file)

            chunks = data["chunks"]

            table_data = []

            for chunk in chunks:

                table_data.append({
                    "Chunk": chunk["chunk_id"],
                    "Timestamp": chunk["timestamp"],
                    "Deepfake": f"{chunk['deepfake_probability'] * 100:.0f}%",
                    "Speaker Match":
                        f"{chunk['speaker_match_probability'] * 100:.0f}%",
                    "Context Risk":
                        f"{chunk['context_risk'] * 100:.0f}%",
                    "Final Risk":
                        f"{chunk['final_risk_score'] * 100:.0f}%",
                    "Reason":
                        ", ".join(chunk["reason_codes"])
                })

            st.dataframe(
                table_data,
                use_container_width=True
            )

        except FileNotFoundError:

            st.info(
                "Sample chunk data is not available."
            )


        # -----------------------------------
        # Reason Codes
        # -----------------------------------

        st.divider()

        st.subheader("Reason Codes")

        st.write(
            "• **SYNTHETIC_VOICE_CHARACTERISTICS** — "
            "Potential synthetic voice characteristics detected."
        )

        st.write(
            "• **SPEAKER_MISMATCH** — "
            "Speaker similarity is below the expected threshold."
        )

        st.write(
            "• **HIGH_RISK** — "
            "Combined signals indicate elevated impersonation risk."
        )

else:

    st.info(
        "Upload a call recording to begin analysis."
    )