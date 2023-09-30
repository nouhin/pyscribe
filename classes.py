import logging
import os
import time

import ffmpeg
import numpy as np
import torch
import whisper

source_folder = 'video_input'
output_folder = 'subs_output'
model = 'medium'
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
    def __init__(self, output_folder=output_folder, language=language, device=device, model_name=model):
        self.output_folder = output_folder
        self.language = language
        self.device = device
        self.model_name = model_name

    def init_model(self):
        self.model = whisper.load_model(name=self.model_name, device=self.device, download_root='models')

    def transcribe(self, audio_file):
        tic = time.time()
        res = self.whisper_model.transcribe(
            audio_file, task="transcribe", language=self.language, verbose=True, word_timestamps=True
        )
        logging.info(f"Done transcription in {time.time() - tic:.1f} sec")
        return res


# Create video class
class Video:
    def __init__(self, path, output_path='None'):
        self.path = path
        self.output_path = output_path
        self.path = path
        self.audio_sample_rate = 16000
        self.audio = None
        self.subtitles = None

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
            self.subtitles = transcriber.transcribe(self.audio)
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
        with open(self.output_path, "w") as f:
            f.write(self.subtitles)
        logging.info(f"Saved subtitles for {self.path}")


def main():

    # Init Whisperer
    whisperer = Whisperer()
    whisperer.init_model()

    # get list of videos
    videos = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]
    logging.info(f"Found {len(videos)} videos to process")

    # Process videos
    for video in videos:
        tic = time.time()
        logging.info(f"Processing video {video}")
        video_path = os.path.join(source_folder, video)
        output_path = os.path.join(output_folder, video.replace('.mp4', '.srt'))
        video = Video(video_path, output_path)
        video.generate_subtitles(whisperer)
        logging.info(f"Processed video {video} in {time.time() - tic:.1f} sec")
