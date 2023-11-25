import gradio as gr
from TTS.api import TTS
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # mps doesn't work yet
else:
    device = "cpu"

torch.set_default_device(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)

def predict(prompt, audio_file_pth, agree):
    if agree == True:
        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=audio_file_pth,
            language="en,
        )

        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")


title = "Coqui🐸 XTTS"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="keep it short!",
            value="LLMs hold the key to generative AI, but some are more suited than others to specific tasks. Here's a guide to the five most powerful and how to use them.",
        ),

        gr.Audio(
            label="Reference Audio",
            info="",
            type="filepath",
            value="examples/male.wav",
        ),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
).queue().launch(debug=True)
