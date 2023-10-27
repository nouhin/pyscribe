import json
import logging
import time
from pathlib import Path

import ffmpeg
import numpy as np
import openai
import tiktoken
import torch
import whisper
import whisper.utils

# # Constants
# SOURCE_FOLDER = Path('video_input')
# OUTPUT_FOLDER = Path('subs_out')
# MODEL = 'medium'
# LANGUAGE = 'french'
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #####################################################################
# print('Current configuration:')
# print(f'Source folder: {SOURCE_FOLDER}')
# print(f'Output folder: {OUTPUT_FOLDER}')
# print(f'Model: {MODEL}')
# print(f'Language: {LANGUAGE}')
# print(f'Device: {DEVICE}')


class ChatGPTHAndler():
    """Handler for ChatGPT API calls."""

    PRICING = {
        'gpt-3.5-turbo': {
            'prompt': 0.0015,
            'completion': 0.002
        },
        'gpt-3.5-turbo-16k': {
            'prompt': 0.003,
            'completion': 0.004
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06
        },
        'gpt-4-32k': {
            'prompt': 0.06,
            'completion': 0.12
        },
    }

    def __init__(self, config):
        self.token = config.token
        openai.api_key = self.token
        self.model = config.gpt_model
        self.prompt = config.default_prompt.replace("{lang}", config.language)
        self.output_folder = Path(config.output_folder)
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)

    def token_count(self, messages):
        """Calculate and return the number of tokens for the given messages."""
        encoding = self._get_encoding()
        num_tokens = sum(self._count_tokens(message, encoding) for message in messages)
        num_tokens += 3  # Priming with assistant
        return num_tokens

    def _get_encoding(self):
        """Get the appropriate encoding for the model."""
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            logging.warning("Model not found. Using cl100k_base encoding.")
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, message, encoding):
        """Count the tokens for a single message."""
        tokens_per_message = self._get_tokens_per_message()
        num_tokens = tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_message - 2
        return num_tokens

    def _get_tokens_per_message(self):
        """Get the number of tokens per message based on the model."""
        if "gpt-3.5-turbo" in self.model:
            return 3
        elif "gpt-4" in self.model:
            return 3
        else:
            raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {self.model}.")

    def get_prompt(self, media):
        """Generate a prompt from media subtitles."""
        transcript = media.subtitles['text']
        return self.prompt + transcript

    def process_media(self, media):
        """Process media and generate GPT summary."""
        try:
            messages = [{"role": "user", "content": self.get_prompt(media)}]
            response = openai.ChatCompletion.create(model=self.model, messages=messages)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            with open(self.output_folder / f"{media.path.stem}_gpt_{timestamp}.json", "w") as f:
                json.dump(response, f)

            self._log_summary_and_costs(response, media)
        except Exception as e:
            logging.error(f"Error processing media: {e}")

    def _log_summary_and_costs(self, response, media):
        """Log the summary and costs of the GPT response."""
        print("ChatGPT summary:")
        print(response["choices"][0]["message"]["content"])

        prompt_cost = response['usage']['prompt_tokens'] * self.PRICING[self.model]['prompt'] / 1000
        completion_cost = response['usage']['completion_tokens'] * self.PRICING[self.model]['completion'] / 1000
        total_cost = prompt_cost + completion_cost

        logging.info(f"GPT summary generated for {media.path}")
        logging.info(f"Total cost: {total_cost} dollars")


class Config:
    def __init__(self, config_file=None):
        # Default values
        self.source_folder = "video_input"
        self.output_folder = "subs_out"
        self.model = "medium"
        self.language = "french"
        self.use_cuda = True

        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load(config_file)

    def load(self, config_file):
        with open(config_file, 'r') as file:
            file_config = json.load(file)
            for key, value in file_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:  # add new key
                    self.__dict__[key] = value

    def save(self, config_file):
        with open(config_file, 'w') as file:
            config_data = {key: getattr(self, key) for key in self.__dict__}
            json.dump(config_data, file, indent=4)

    def print_config(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')


class Whisperer:
    def __init__(self, config) -> None:
        self.language = config.language
        self.device = config.device
        self.model_name = config.model
        self.model = None
        self.init_model()

    def init_model(self):
        logging.info(f"Loading model {self.model_name} on {self.device}")
        self.model = whisper.load_model(name=self.model_name, device=self.device, download_root='models')
        logging.info(f"Loaded model {self.model_name} on {self.device}")

    def transcribe(self, audio_file, context=None) -> dict[str, str | list]:
        if self.model is None:
            logging.error("Model not initialized")
            raise RuntimeError("Model not initialized")
        tic = time.time()
        options = {
            "task": "transcribe",
            "language": self.language,
            "verbose": True,
            "word_timestamps": True,
            "fp16": False
        }
        if context:
            options["initial_prompt"] = context
        result = self.model.transcribe(audio_file, **options)
        logging.info(f"Completed transcription in {time.time() - tic:.1f} sec")
        return result


class Media:
    def __init__(self, path, config, r_out=None, context=None, save_raw=False):
        self.path = Path(path)
        self.output_path = Path(config.output_folder)
        self.raw_output_path = Path(r_out) if r_out else self.output_path
        self.save_raw_flag = save_raw
        self.audio_sample_rate = 16000
        self.audio = None
        self.subtitles = None
        self.context = context

    def get_audio(self):
        logging.info(f"Extracting audio from {self.path}")
        try:
            out, _ = ffmpeg.input(self.path, threads=0).output(
                "-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.audio_sample_rate
            ).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            self.audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        except ffmpeg.Error as e:
            logging.error(f"Failed to extract audio: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}") from e

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
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        raw_output_path = self.raw_output_path / f"{self.path.stem}_raw_{timestamp}.json"
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
        COLORS = {logging.ERROR: ('\033[91m', '\033[0m')}

        def format(self, record):
            color_prefix, color_suffix = self.COLORS.get(record.levelno, ('', ''))
            record.msg = f"{color_prefix}{record.msg}{color_suffix}"
            return super().format(record)

    console_handler = ColorizedHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('pyscribe.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Usage Example
# # Setup logging
# setup_logging()

# logging.info("This is an info message.")
# logging.error("This is an error message.")

# # Initialize configuration
# config = Config('config.json')  # Assuming a file named 'config.json'
# config.print_config()

# # read context from file
# context_file = Path('context.txt')
# context = context_file.read_text() if context_file.exists() else None

# # Initialize Whisperer and Media with configuration
# whisperer = Whisperer(config)
# video_file = "example.mp4"
# media = Media(path=video_file, config=config, context=context, save_raw=True)
# tic = time.time()
# media.generate_subtitles(whisperer)
# logging.info(f"Processed video {video_file} in {time.time() - tic:.1f} sec")

if __name__ == "__main__":
    setup_logging()

    logging.info("This is an info message.")
    logging.error("This is an error message.")

    config = Config()
    config.load('key.json')
    config.load('config.json')

    config.print_config()

    context_file = Path('context.txt')
    context = context_file.read_text() if context_file.exists() else None

    # Init Whisperer and gpt
    whisperer = Whisperer(config)
    gpt = ChatGPTHAndler(config)

    # get list of videos in source folder, extension .mp4 mkv
    source_folder = Path(config.source_folder)
    videos = [f for f in source_folder.iterdir() if f.is_file() and f.suffix in [".mp4", ".mkv"]]
    logging.info(f"Found {len(videos)} videos to process")

    # Process videos
    for video in videos:
        tic = time.time()
        logging.info(f"Processing video {video.name}")
        video_obj = Media(path=video, config=config, context=context, save_raw=True)
        video_obj.generate_subtitles(whisperer)
        gpt.process_media(video_obj)
        logging.info(f"Processed video {video.name} in {time.time() - tic:.1f} sec")
