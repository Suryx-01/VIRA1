import os


def get_audio_info(audio_file):
    """
    Extract basic information about an uploaded audio file.

    This is a Day-1 placeholder.
    Actual audio preprocessing will be added when the
    real AI models are integrated.
    """

    file_name = os.path.basename(audio_file.name)

    file_size = len(audio_file.getvalue())

    return {
        "file_name": file_name,
        "file_size_bytes": file_size,
        "status": "Audio received successfully"
    }