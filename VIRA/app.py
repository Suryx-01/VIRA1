import os
import json
import tempfile

import streamlit as st

from backend.audio_processor import get_audio_info
from backend.deepfake_detector import detect_deepfake
from backend.speaker_verifier import verify_speaker
from backend.risk_engine import calculate_risk


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VIRA Voice Risk Dashboard",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #F8FAFC;
    }

    .panel {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .risk-gauge {
        background-color: #111827;
        border: 2px solid #EF4444;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
    }

    .risk-number {
        font-size: 52px;
        font-weight: bold;
    }

    .timeline-low {
        border-left: 5px solid #22C55E;
        background-color: #111827;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }

    .timeline-medium {
        border-left: 5px solid #F59E0B;
        background-color: #111827;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }

    .timeline-high {
        border-left: 5px solid #EF4444;
        background-color: #111827;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }

    .reason {
        background-color: #111827;
        border-left: 4px solid #EF4444;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ VIRA Voice Risk Dashboard")

st.caption(
    "Call ID: VIRA-DEMO-001 | "
    "Claimed Speaker: Speaker A | "
    "Mode: Simulated Live | "
    "Chunk: 5s"
)

st.divider()


# ============================================================
# LOAD SAMPLE CHUNK DATA
# ============================================================

try:

    with open("sample_chunk_scores.json", "r") as file:
        data = json.load(file)

    chunks = data.get("chunks", [])
    call_id = data.get("call_id", "VIRA-DEMO-001")

except Exception:

    chunks = []
    call_id = "VIRA-DEMO-001"


# ============================================================
# DEFAULT VALUES
# ============================================================

deepfake_probability = 0.0
speaker_match_probability = 0.0
context_risk = 0.0
uncertainty = 0.35

final_risk = 0.0
current_action = "Allow"

speaker_result = None
deepfake_result = None
risk_result = None

analysis_completed = False


# ============================================================
# THREE MAIN PANELS
# ============================================================

left, center, right = st.columns([1, 2, 1])


# ============================================================
# LEFT PANEL
# ============================================================

with left:

    st.subheader("Input")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "ogg", "m4a"]
    )

    st.subheader("Claimed Speaker")

    claimed_speaker = st.selectbox(
        "Select Speaker",
        [
            "Speaker A",
            "Speaker B",
            "Speaker C",
            "Unknown"
        ]
    )

    st.subheader("Context Flags")

    urgent = st.checkbox("Urgent request")
    financial = st.checkbox("Financial context")
    sensitive = st.checkbox("Sensitive information")
    unusual = st.checkbox("Unusual behaviour")

    st.write("")

    analyze = st.button(
        "🔍 Analyze Call",
        type="primary",
        use_container_width=True
    )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

if audio_file is not None:

    # --------------------------------------------------------
    # Display basic audio information
    # --------------------------------------------------------

    try:

        audio_info = get_audio_info(audio_file)

        with st.expander("Audio Information"):

            st.write(audio_info)

    except Exception as e:

        st.warning(
            f"Could not read audio information: {e}"
        )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if analyze:

        st.info("Analyzing uploaded audio...")

        temp_audio_path = None

        try:

            # ====================================================
            # SAVE STREAMLIT UPLOAD TO TEMPORARY FILE
            # ====================================================

            file_extension = os.path.splitext(
                audio_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    audio_file.getbuffer()
                )

                temp_audio_path = temp_file.name


            # ====================================================
            # DEEPFAKE DETECTION
            # ====================================================

            st.write("Running deepfake detection...")

            deepfake_result = detect_deepfake(
                audio_file
            )

            deepfake_probability = float(
                deepfake_result.get(
                    "deepfake_probability",
                    0.0
                )
            )


            # ====================================================
            # SPEAKER VERIFICATION
            # ====================================================

            st.write(
                "Running TitaNet + ECAPA speaker verification..."
            )

            speaker_result = verify_speaker(
                temp_audio_path
            )

            speaker_match_probability = float(
                speaker_result.get(
                    "speaker_match_probability",
                    0.0
                )
            )


            # ====================================================
            # CONTEXT RISK
            # ====================================================

            context_risk = 0.0

            if urgent:
                context_risk += 0.20

            if financial:
                context_risk += 0.30

            if sensitive:
                context_risk += 0.20

            if unusual:
                context_risk += 0.20

            context_risk = min(
                context_risk,
                1.0
            )


            # ====================================================
            # RISK ENGINE
            # ====================================================

            risk_result = calculate_risk(
                deepfake_probability,
                speaker_match_probability,
                context_risk
            )

            final_risk = float(
                risk_result.get(
                    "final_risk_score",
                    0.0
                )
            )

            current_action = risk_result.get(
                "recommended_action",
                "Allow"
            )


            # ====================================================
            # ANALYSIS COMPLETE
            # ====================================================

            analysis_completed = True

            st.success(
                "Audio analysis completed successfully."
            )


        except Exception as e:

            st.error(
                "Analysis failed."
            )

            st.exception(e)


        finally:

            # ----------------------------------------------------
            # Delete temporary file
            # ----------------------------------------------------

            if (
                temp_audio_path is not None
                and os.path.exists(temp_audio_path)
            ):

                try:
                    os.remove(temp_audio_path)

                except Exception:
                    pass


# ============================================================
# CENTER PANEL
# ============================================================

with center:

    st.subheader("Live Risk Timeline")

    if chunks:

        for chunk in chunks:

            chunk_risk_result = calculate_risk(
                chunk["deepfake_probability"],
                chunk["speaker_match_probability"],
                chunk["context_risk"]
            )

            chunk_risk = chunk_risk_result[
                "final_risk_score"
            ]

            chunk_action = chunk_risk_result[
                "recommended_action"
            ]


            if chunk_risk >= 0.80:

                status = "🔴 HIGH"
                css_class = "timeline-high"

            elif chunk_risk >= 0.50:

                status = "🟡 MEDIUM"
                css_class = "timeline-medium"

            else:

                status = "🟢 LOW"
                css_class = "timeline-low"


            st.markdown(
                f"""
                <div class="{css_class}">
                    <b>Chunk {chunk['chunk_id']}</b>
                    &nbsp; | &nbsp;
                    {chunk['timestamp']}
                    &nbsp; | &nbsp;
                    {status}
                    &nbsp; | &nbsp;
                    Risk: {chunk_risk * 100:.0f}%
                    &nbsp; | &nbsp;
                    Action: {chunk_action}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "No chunk data available."
        )


    # ========================================================
    # RISK GAUGE
    # ========================================================

    st.subheader("Risk Gauge")

    st.metric(
        "FINAL RISK SCORE",
        f"{final_risk * 100:.0f}%"
    )

    st.progress(
        min(max(final_risk, 0.0), 1.0)
    )


    # ========================================================
    # CURRENT ACTION
    # ========================================================

    st.subheader("Current Action")

    if final_risk >= 0.80:

        st.error(
            f"🚨 {current_action}"
        )

    elif final_risk >= 0.50:

        st.warning(
            f"⚠️ {current_action}"
        )

    else:

        st.success(
            f"✅ {current_action}"
        )


# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.subheader("Model Scores")

    st.metric(
        "Deepfake",
        f"{deepfake_probability * 100:.0f}%"
    )

    st.metric(
        "Speaker Match",
        f"{speaker_match_probability * 100:.0f}%"
    )

    st.metric(
        "Context Risk",
        f"{context_risk * 100:.0f}%"
    )

    st.metric(
        "Uncertainty",
        f"{uncertainty * 100:.0f}%"
    )


    # ========================================================
    # SPEAKER MODEL DETAILS
    # ========================================================

    if speaker_result is not None:

        st.subheader("Speaker Verification")

        titanet_similarity = speaker_result.get(
            "titanet_similarity"
        )

        ecapa_similarity = speaker_result.get(
            "ecapa_similarity"
        )

        speaker_reason = speaker_result.get(
            "reason_code"
        )

        if titanet_similarity is not None:

            st.metric(
                "TitaNet Similarity",
                f"{float(titanet_similarity) * 100:.1f}%"
            )

        if ecapa_similarity is not None:

            st.metric(
                "ECAPA Similarity",
                f"{float(ecapa_similarity) * 100:.1f}%"
            )

        if speaker_reason:

            st.write(
                f"Decision: **{speaker_reason}**"
            )


    # ========================================================
    # REASON CODES
    # ========================================================

    st.subheader("Reason Codes")

    reason_codes = []


    if deepfake_probability >= 0.50:

        reason_codes.append(
            "synthetic artifacts"
        )


    if context_risk >= 0.30:

        reason_codes.append(
            "financial / high-risk context"
        )


    if speaker_match_probability < 0.70:

        reason_codes.append(
            "identity uncertainty"
        )


    if not reason_codes:

        reason_codes.append(
            "no major risk indicators"
        )


    for reason in reason_codes:

        st.markdown(
            f"""
            <div class="reason">
                • {reason}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ANALYSIS RESULT DETAILS
# ============================================================

if analysis_completed:

    st.divider()

    st.subheader("Current Audio Analysis")

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Deepfake Score",
            f"{deepfake_probability * 100:.1f}%"
        )


    with result_col2:

        st.metric(
            "Speaker Match",
            f"{speaker_match_probability * 100:.1f}%"
        )


    with result_col3:

        st.metric(
            "Final Risk",
            f"{final_risk * 100:.1f}%"
        )


    # ========================================================
    # RAW MODEL RESULTS
    # ========================================================

    with st.expander("View Model Results"):

        if deepfake_result is not None:

            st.write(
                "### Deepfake Detector"
            )

            st.json(
                deepfake_result
            )


        if speaker_result is not None:

            st.write(
                "### Speaker Verification"
            )

            st.json(
                speaker_result
            )


        if risk_result is not None:

            st.write(
                "### Risk Engine"
            )

            st.json(
                risk_result
            )


# ============================================================
# BOTTOM CHUNK TABLE
# ============================================================

st.divider()

st.subheader(
    "Chunk-by-Chunk Analysis"
)

if chunks:

    table_data = []

    for chunk in chunks:

        chunk_risk_result = calculate_risk(
            chunk["deepfake_probability"],
            chunk["speaker_match_probability"],
            chunk["context_risk"]
        )

        chunk_risk = chunk_risk_result[
            "final_risk_score"
        ]

        chunk_action = chunk_risk_result[
            "recommended_action"
        ]


        table_data.append(
            {
                "Chunk":
                    chunk["chunk_id"],

                "Time":
                    chunk["timestamp"],

                "Deepfake":
                    f"{chunk['deepfake_probability'] * 100:.0f}%",

                "Speaker Match":
                    f"{chunk['speaker_match_probability'] * 100:.0f}%",

                "Context":
                    f"{chunk['context_risk'] * 100:.0f}%",

                "Risk":
                    f"{chunk_risk * 100:.0f}%",

                "Action":
                    chunk_action,

                "Reasons":
                    ", ".join(
                        chunk.get(
                            "reason_codes",
                            []
                        )
                    )
            }
        )


    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Chunk data unavailable."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VIRA MVP • Simulated Live Mode • "
    "Deepfake detection and TitaNet + ECAPA "
    "speaker verification are integrated."
)