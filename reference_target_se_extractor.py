import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
print(device)

tone_color_converter = ToneColorConverter(
    f'{ckpt_converter}/config.json', device=device)

m = tone_color_converter.model
model_million_params = sum(p.numel() for p in m.parameters())/1e6
print(m)
print(f"{model_million_params}M parameters")

tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# This is the voice you want to clone
reference_speaker = 'resources/me_reference.mp3'
# 提取目标说话者的音色
target_dir = "processed"
target_se, audio_name = se_extractor.get_se(
    reference_speaker, tone_color_converter, target_dir=target_dir, vad=True)
se_path = os.path.join(target_dir, audio_name, 'se.pth')
print(f"saved target tone color file: {se_path} ")
