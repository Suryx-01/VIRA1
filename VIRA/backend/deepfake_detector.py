from scipy.signal import resample_poly
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_SAMPLES = 64600

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml


# ============================================================
# FIND RAWGAT-ST PROJECT
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# VIRA/backend/deepfake_detector.py
#        ↑
# Go up twice to the parent folder containing VIRA and RawGAT-ST
PROJECTS_DIR = CURRENT_FILE.parents[2]

RAWGAT_DIR = PROJECTS_DIR / "RawGAT-ST-antispoofing"

if not RAWGAT_DIR.exists():
    raise FileNotFoundError(
        f"RawGAT-ST folder not found:\n{RAWGAT_DIR}"
    )

# Allow Python to import the original RawGAT-ST files
sys.path.insert(0, str(RAWGAT_DIR))


from model import RawGAT_ST


# ============================================================
# RAWGAT-ST FILES
# ============================================================

CONFIG_PATH = RAWGAT_DIR / "model_config_RawGAT_ST.yaml"

MODEL_PATH = (
    RAWGAT_DIR
    / "Pre_trained_models"
    / "RawGAT_ST_mul"
    / "Best_epoch.pth"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXPECTED_SAMPLES = 64600
EXPECTED_SAMPLE_RATE = 16000


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def _load_model():
    global _model

    if _model is not None:
        return _model

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"RawGAT-ST config not found:\n{CONFIG_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RawGAT-ST checkpoint not found:\n{MODEL_PATH}"
        )

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]

    model = RawGAT_ST(model_config, DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    _model = model

    return _model


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def _prepare_audio(audio_file):
    """
    Convert the uploaded audio into the format expected
    by RawGAT-ST.
    """

    # Streamlit UploadedFile
    if hasattr(audio_file, "getvalue"):
        audio_bytes = audio_file.getvalue()

        import io

        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes)
        )

    else:
        # Normal file path
        audio, sample_rate = sf.read(audio_file)

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Convert audio to 16 kHz if necessary.
    if sample_rate != EXPECTED_SAMPLE_RATE:
        gcd = np.gcd(sample_rate, EXPECTED_SAMPLE_RATE)

        up = EXPECTED_SAMPLE_RATE // gcd
        down = sample_rate // gcd

        audio = resample_poly(
            audio,
            up,
            down
        ).astype(np.float32)
        
    # RawGAT-ST uses 64600 samples.
    if len(audio) == 0:
        raise ValueError("Audio file contains no audio samples.")

    if len(audio) >= EXPECTED_SAMPLES:
        audio = audio[:EXPECTED_SAMPLES]

    else:
        repeats = int(
            EXPECTED_SAMPLES / len(audio)
        ) + 1

        audio = np.tile(
            audio,
            repeats
        )[:EXPECTED_SAMPLES]

    return audio


# ============================================================
# DEEPFAKE DETECTION
# ============================================================

def detect_deepfake(audio_file):

    model = _load_model()

    audio = _prepare_audio(audio_file)

    # Convert to PyTorch tensor
    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # RawGAT-ST inference
    with torch.no_grad():

        output = model(
            audio_tensor,
            Freq_aug=False
        )

    # RawGAT-ST repository uses class 1 output as
    # its spoof/deepfake score.
    raw_score = float(
        output[:, 1].detach().cpu().numpy().ravel()[0]
    )

    # Convert model output into a 0-1 probability-like value
    # for VIRA's existing interface.
    deepfake_probability = float(
        torch.sigmoid(
            torch.tensor(raw_score)
        ).item()
    )

    prediction = (
        "FAKE"
        if deepfake_probability >= 0.5
        else "BONAFIDE"
    )

    return {
        "deepfake_probability": deepfake_probability,
        "prediction": prediction,
        "model": "RawGAT-ST",
        "raw_score": raw_score,
    }