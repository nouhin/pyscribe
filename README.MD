# Pyscribe

Simple tool that transcribes videos using OpenAI's whisper.

I will update this readme with instructions on setup and usage (maybe).

Sample config.json :

```json
{
    "source_folder": "video_input",
    "output_folder": "subs_out",
    "model": "medium",
    "language": "french",
    "use_cuda": true,
    "default_prompt": "Summarize the following text, making it easy to read and comprehend. The summary should be concise, clear, and capture the main points of the text. Avoid using complex sentence structures. Respond in {lang}. Please begin by editing the following text: ",
    "gpt_model": "gpt-4"
}
```
Sample key.json :

```json
{
    "token": "YOUR_TOKEN_HERE"
}
```