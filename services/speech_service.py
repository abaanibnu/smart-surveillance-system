import os
import shutil

try:
    import imageio_ffmpeg
    # Ensure ffmpeg binary is accessible as 'ffmpeg.exe' on PATH for Whisper (Windows)
    _ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_src)
    _ffmpeg_dst = os.path.join(_ffmpeg_dir, "ffmpeg.exe")

    if not os.path.exists(_ffmpeg_dst):
        try:
            shutil.copy(_ffmpeg_src, _ffmpeg_dst)
        except Exception as e:
            pass

    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import whisper

class VoiceQueryHandler:
    """
    Implements Voice Search Interaction via OpenAI Whisper (DFD Level 1.5 & Sequence Diagram 5.14).
    Accepts operator audio input (WAV/WEBM/MP3/OGG) and transcribes to natural-language text.
    """

    def __init__(self, model_name="base.en"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribes an audio recording into text.
        Returns: { "success": bool, "text": str, "language": str }
        """
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"Audio file not found: {audio_path}", "text": ""}

        try:
            model = self._get_model()
            result = model.transcribe(audio_path, fp16=False)
            text = result.get("text", "").strip()
            language = result.get("language", "en")
            
            return {
                "success": True,
                "text": text,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Speech transcription error: {str(e)}",
                "text": ""
            }
