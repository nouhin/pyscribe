import json
import logging
import os
import time

import ffmpeg
import numpy as np
import torch
import whisper

source_folder = 'video_input'
output_folder = 'subs_out'
model = 'small'
language = 'french'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################

print('Current configuration:')
print('Source folder: {}'.format(source_folder))
print('Output folder: {}'.format(output_folder))
print('Model: {}'.format(model))
print('Language: {}'.format(language))
print('Device: {}'.format(device))


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
                initial_prompt=context
            )
        else:
            res = self.model.transcribe(
                audio_file, task="transcribe", language=self.language, verbose=True, word_timestamps=True
            )
        logging.info(f"Done transcription in {time.time() - tic:.1f} sec")
        return res


# Create video class
class Video:
    def __init__(self, path, output_path=None, context=None):
        self.path = path
        self.output_path = output_path
        self.path = path
        self.audio_sample_rate = 16000
        self.audio = None
        self.subtitles: dict[str, str | list]
        self.context: str | None = context

    def get_audio(self):
        logging.info(f"Loading audio from {self.path}")
        try:
            out, _ = (
                ffmpeg.input(self.path, threads=0).output(
                    "-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.audio_sample_rate
                ).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logging.error(f"Failed to load audio: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        self.audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        logging.info(f"Loaded audio from {self.path}")

    def generate_subtitles(self, transcriber):
        logging.info(f"Generating subtitles for {self.path}")
        if self.audio is None:
            self.get_audio()
        try:
            self.subtitles = transcriber.transcribe(self.audio, context=self.context)
            self.save_subtitles()
            logging.info(f"Generated subtitles for {self.path}")
        except Exception as e:
            logging.error(f"Failed to generate subtitles: {e}")
            raise RuntimeError(f"Failed to generate subtitles: {e}") from e

    def save_subtitles(self):
        logging.info(f"Saving subtitles for {self.path}")
        if self.subtitles is None:
            logging.error("No subtitles to save")
            raise RuntimeError("No subtitles to save")
        if self.output_path is None:
            self.output_path = 'subtitle_output'
        # create output folder if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            # serialize subtitles with json
            json.dump(self.subtitles, f, indent=4)
        logging.info(f"Saved subtitles for {self.path}")


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

    # get list of videos
    videos = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]
    logging.info(f"Found {len(videos)} videos to process")

    # read context from file
    try:
        with open('context.txt', 'r') as f:
            context = f.read()
    except FileNotFoundError:
        context = None

    # Process videos
    for video in videos:
        tic = time.time()
        logging.info(f"Processing video {video}")
        video_path = os.path.join(source_folder, video)
        output_path = os.path.join(output_folder, video.replace('.mp4', '.srt'))
        video = Video(video_path, output_path)
        video.context = context
        video.generate_subtitles(whisperer)
        logging.info(f"Processed video {video} in {time.time() - tic:.1f} sec")


if __name__ == "__main__":
    main()
