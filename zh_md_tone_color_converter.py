from melo.api import TTS
import os
import re
import torch
from openvoice.api import ToneColorConverter
import argparse


def filter_md_special_tag(context):
    meta_blocks_pattern = r'^---[\s\S]*?---'

    code_blocks_pattern = r'```[\s\S]*?```'
    code_block_pattern = r'`[^`]+`'

    img_block_pattern = r'!\[.*?\]\(.*?\)'
    audio_block_pattern = r'\[.*?\]\(.*?.(?:mp3|ogg|wav|mp4)\)'

    html_blocks_pattern = r'<!--[\s\S]*?-->'
    hugo_tag_blocks_pattern = r'{{[\s\S]*?}}'

    ref_tag_blocks_pattern = r'\[\^\d+\]:.*?$'

    text = re.sub(meta_blocks_pattern, '', context)
    text = re.sub(code_blocks_pattern, '', text)
    text = re.sub(code_block_pattern, '', text)
    text = re.sub(img_block_pattern, '', text)
    text = re.sub(audio_block_pattern, '', text)
    text = re.sub(html_blocks_pattern, '', text)
    text = re.sub(hugo_tag_blocks_pattern, '', text)
    text = re.sub(ref_tag_blocks_pattern, '', text, flags=re.MULTILINE)

    return text.strip() + "。\n文章阅读结束,感谢收听。"


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cc", "--ckpt_converter_dir", type=str,
                        default='checkpoints_v2/converter', help="")
    parser.add_argument("-cbss", "--ckpt_base_speakers_ses_dir", type=str,
                        default='checkpoints_v2/base_speakers/ses', help="")
    parser.add_argument("-od", "--output_dir", type=str,
                        default='outputs_v2', help="")
    parser.add_argument("-tsp", "--target_se_path", type=str,
                        default='processed/me_reference_v2_2oULQ_^lIO3MSGW77/se.pth', help="")
    parser.add_argument("-t", "--text", type=str,
                        default='hello world', help="")
    parser.add_argument("-tf", "--text_file", type=str,
                        default='', help="")
    parser.add_argument("-sn", "--save_name", type=str,
                        default='output_v2_zh', help="")
    args = parser.parse_args()
    print(f'args: {args}')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    ckpt_converter_dir = args.ckpt_converter_dir
    output_dir = args.output_dir
    target_se_path = args.target_se_path
    ckpt_base_speakers_ses_dir = args.ckpt_base_speakers_ses_dir
    text = args.text
    if len(args.text_file) > 0:
        with open(args.text_file, 'r') as f:
            text = f.read()
            text = filter_md_special_tag(text)
    save_name = args.save_name
    print(text)

    # load tone_color_converter ckpt
    tone_color_converter = ToneColorConverter(
        f'{ckpt_converter_dir}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter_dir}/checkpoint.pth')

    # load target se
    target_se = torch.load(target_se_path, map_location=device)
    # print(target_se)

    src_path = f'{output_dir}/tmp.wav'
    # Speed is adjustable
    speed = 1.0

    model = TTS(language='ZH', device=device)
    speaker_ids = model.hps.data.spk2id
    speaker_id = speaker_ids['ZH']

    # 使用meloTTS作为基础原始模型将文本转换成语音,将音频文件保存在src_path中
    source_se = torch.load(
        f'{ckpt_base_speakers_ses_dir}/zh.pth', map_location=device)
    model.tts_to_file(text, speaker_id, src_path, speed=speed)

    save_path = f'{output_dir}/{save_name}.wav'
    # 将源音频和目标音频进行音色转换, 并将转换后的音频文件保存到save_path中
    encode_message = "@Weedge"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message)
