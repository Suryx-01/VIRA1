import os

import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F

import nemo.collections.asr as nemo_asr

from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.speaker import SpeakerRecognition


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_AUDIO = r"C:\Users\ASUS\Downloads\REAL_01_English_S01.wav"

TARGET_SAMPLE_RATE = 16000
SPEAKER_THRESHOLD = 0.70

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL CACHE
# ============================================================

_titanet_model = None
_ecapa_model = None


# ============================================================
# LOAD TITANET
# ============================================================

def _load_titanet():

    global _titanet_model

    if _titanet_model is not None:
        return _titanet_model

    print("Loading TitaNet-L...")

    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        model_name="titanet_large"
    )

    model = model.to(DEVICE)
    model.eval()

    print("TitaNet-L loaded successfully.")

    _titanet_model = model

    return _titanet_model


# ============================================================
# LOAD ECAPA
# ============================================================

def _load_ecapa():

    global _ecapa_model

    if _ecapa_model is not None:
        return _ecapa_model

    print("Loading ECAPA-TDNN...")

    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_ecapa",
        local_strategy=LocalStrategy.COPY
    )

    print("ECAPA-TDNN loaded successfully.")

    _ecapa_model = model

    return _ecapa_model


# ============================================================
# LOAD AUDIO FOR TITANET
# ============================================================

def _load_audio(audio_file):

    audio, sample_rate = sf.read(
        audio_file,
        dtype="float32",
        always_2d=False
    )

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio)

    # Resample
    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            TARGET_SAMPLE_RATE
        )

    # [time] -> [batch, time]
    waveform = waveform.unsqueeze(0)

    waveform = waveform.to(DEVICE)

    return waveform


# ============================================================
# TITANET EMBEDDING
# ============================================================

@torch.no_grad()
def _get_titanet_embedding(audio_file):

    model = _load_titanet()

    waveform = _load_audio(audio_file)

    audio_length = torch.tensor(
        [waveform.shape[1]],
        dtype=torch.long,
        device=DEVICE
    )

    _, embedding = model(
        input_signal=waveform,
        input_signal_length=audio_length
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


# ============================================================
# ECAPA EMBEDDING
# ============================================================

@torch.no_grad()
def _get_ecapa_embedding(audio_file):

    model = _load_ecapa()

    # Use the same audio preprocessing as TitaNet
    waveform = _load_audio(audio_file)

    # ECAPA expects [batch, time]
    # Our _load_audio() already returns [1, time]

    embedding = model.encode_batch(
        waveform
    )

    # Normalize embedding
    embedding = F.normalize(
        embedding,
        p=2,
        dim=-1
    )

    return embedding


# ============================================================
# ECAPA SPEAKER SIMILARITY
# ============================================================

def _get_ecapa_score(reference_audio, test_audio):

    reference_embedding = _get_ecapa_embedding(
        reference_audio
    )

    test_embedding = _get_ecapa_embedding(
        test_audio
    )

    similarity = F.cosine_similarity(
        reference_embedding.reshape(1, -1),
        test_embedding.reshape(1, -1)
    ).item()

    return float(similarity)

# ============================================================
# SPEAKER VERIFICATION
# ============================================================

def verify_speaker(audio_file):

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(REFERENCE_AUDIO):

        raise FileNotFoundError(
            f"Reference speaker audio not found:\n"
            f"{REFERENCE_AUDIO}"
        )

    if not os.path.exists(audio_file):

        raise FileNotFoundError(
            f"Test audio file not found:\n"
            f"{audio_file}"
        )

    print("\nRunning speaker verification...")
    print("Reference:", REFERENCE_AUDIO)
    print("Test:", audio_file)


    # ========================================================
    # TITANET
    # ========================================================

    reference_embedding = _get_titanet_embedding(
        REFERENCE_AUDIO
    )

    test_embedding = _get_titanet_embedding(
        audio_file
    )

    titanet_similarity = F.cosine_similarity(
        reference_embedding,
        test_embedding
    ).item()


    # ========================================================
    # ECAPA
    # ========================================================

    ecapa_score = _get_ecapa_score(
        REFERENCE_AUDIO,
        audio_file
    )

    ecapa_prediction = (
        1 if ecapa_score >= SPEAKER_THRESHOLD else 0
    )


    # ========================================================
    # FINAL SPEAKER SCORE
    # ========================================================

    # IMPORTANT:
    #
    # These scores are similarity scores, NOT calibrated
    # probabilities.
    #
    # For now we keep them separate and use their average
    # as a combined display score.
    #
    # We can calibrate this properly later using validation
    # data.

    combined_score = (
        titanet_similarity + ecapa_score
    ) / 2.0

    combined_score = max(
        0.0,
        min(1.0, combined_score)
    )


    # ========================================================
    # DECISION
    # ========================================================

    if combined_score >= SPEAKER_THRESHOLD:

        reason_code = "SPEAKER_MATCH"

    else:

        reason_code = "SPEAKER_MISMATCH"


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "speaker_match_probability":
            combined_score,

        "titanet_similarity":
            titanet_similarity,

        "ecapa_similarity":
            ecapa_score,

        "ecapa_prediction":
            ecapa_prediction,

        "reason_code":
            reason_code,

        "status":
            "TitaNet + ECAPA speaker verification completed"
    }


    print("\nSpeaker verification result:")
    print(result)

    return result