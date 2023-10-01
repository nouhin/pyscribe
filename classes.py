import json
import logging
import time
from pathlib import Path

import ffmpeg
import numpy as np
import torch
import whisper
import whisper.utils

source_folder = Path('video_input')
output_folder = Path('subs_out')
model = 'medium'
language = 'french'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################

print('Current configuration:')
print(f'Source folder: {source_folder}')
print(f'Output folder: {output_folder}')
print(f'Model: {model}')
print(f'Language: {language}')
print(f'Device: {device}')


class Whisperer:
    def __init__(self, output_folder=output_folder, language=language, device=device, model_name=model) -> None:
        self.output_folder = output_folder
        self.language = language
        self.device = device
        self.model_name = model_name
        self.model: whisper.Whisper

    def init_model(self) -> None:
        logging.info(f"Loading model {self.model_name} on {self.device}")
        self.model = whisper.load_model(name=self.model_name, device=self.device, download_root='models')
        logging.info(f"Loaded model {self.model_name} on {self.device}")

    def transcribe(self, audio_file, context=None) -> dict[str, str | list]:
        if model is None:
            logging.error("Model not initialized")
            raise RuntimeError("Model not initialized")
        tic = time.time()
        if context:
            res = self.model.transcribe(
                audio_file,
                task="transcribe",
                language=self.language,
                verbose=True,
                word_timestamps=True,
                initial_prompt=context,
            )
        else:
            res = self.model.transcribe(
                audio_file, task="transcribe", language=self.language, verbose=True, word_timestamps=True
            )
        logging.info(f"Done transcription in {time.time() - tic:.1f} sec")
        return res


# Create video class
class Video:
    def __init__(self, path, s_out=None, r_out=None, context=None, save_raw=False) -> None:
        self.path: Path = path
        self.output_path: Path = Path(s_out) if s_out else Path('subs_out')
        self.raw_output_path: Path = Path(r_out) if r_out else Path('subs_out')
        self.save_raw_flag = save_raw
        self.audio_sample_rate = 16000
        self.audio = None
        self.subtitles: dict[str, str | list] | None = None
        self.context: str | None = context

    def get_audio(self):
        logging.info(f"Loading audio from {self.path}")
        try:
            out, _ = (
                ffmpeg.input(self.path, threads=0).output(
                    "-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.audio_sample_rate
                ).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            self.audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            logging.info(f"Loaded audio from {self.path}")
        except ffmpeg.Error as e:
            logging.error(f"Failed to load audio: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    def generate_subtitles(self, transcriber):
        logging.info(f"Generating subtitles for {self.path}")
        if self.audio is None:
            self.get_audio()
        try:
            self.subtitles = transcriber.transcribe(self.audio, context=self.context)
            if self.save_raw_flag:
                self.save_raw()
            self.save_subtitles()
            logging.info(f"Generated subtitles for {self.path}")
        except Exception as e:
            logging.error(f"Failed to generate subtitles: {e}")
            raise RuntimeError(f"Failed to generate subtitles: {e}") from e

    def save_raw(self):
        logging.info(f"Saving raw output for {self.path}")
        if self.subtitles is None:
            logging.error("No subtitles to save")
            raise RuntimeError("No subtitles to save")

        raw_output_path = self.raw_output_path / f"{self.path.stem}.json"
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)

        with raw_output_path.open('w') as f:
            json.dump(self.subtitles, f, indent=4)
        logging.info(f"Saved raw output for {self.path}")

    def save_subtitles(self):
        logging.info(f"Saving subtitles for {self.path}")
        if self.subtitles is None:
            logging.error("No subtitles to save")
            raise RuntimeError("No subtitles to save")

        output_path = self.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        srt_writer = whisper.utils.get_writer("srt", str(output_path))

        output_path = self.path.stem + ".srt"

        logging.info(f"Saving subtitles for {self.path} to {output_path}")

        try:
            srt_writer(
                self.subtitles,
                str(output_path),  # type: ignore
                {
                    "max_line_width": 47,
                    "max_line_count": 1,
                    "highlight_words": False
                }
            )
        except Exception as e:
            logging.error(f"Failed to save subtitles: {e}")
            raise RuntimeError(f"Failed to save subtitles: {e}") from e


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    class ColorizedHandler(logging.StreamHandler):
        def format(self, record):
            if record.levelno == logging.ERROR:
                color_prefix = '\033[91m'
                color_suffix = '\033[0m'
                record.msg = f"{color_prefix}{record.msg}{color_suffix}"
            return super().format(record)

    console_handler = ColorizedHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('pyscribe.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main():
    setup_logging()

    logging.info("This is an info message.")
    logging.error("This is an error message.")

    # Init Whisperer
    whisperer = Whisperer()
    whisperer.init_model()

    # get list of videos in source folder, extension .mp4 mkv
    videos = [f for f in source_folder.iterdir() if f.is_file() and f.suffix in ['.mp4', '.mkv']]
    logging.info(f"Found {len(videos)} videos to process")

    # read context from file
    context_file = Path('context.txt')
    context = context_file.read_text() if context_file.exists() else None

    # Process videos
    for video in videos:
        tic = time.time()
        logging.info(f"Processing video {video.name}")
        video_obj = Video(video, save_raw=True)
        video_obj.context = context
        video_obj.generate_subtitles(whisperer)
        logging.info(f"Processed video {video.name} in {time.time() - tic:.1f} sec")


if __name__ == "__main__":
    main()
