# Seed-VC
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)  

*[English](README.md) | 简体中文 | [日本語](README-JP.md)*    

目前发布的模型支持零样本语音转换🔊，零样本实时语音转换🙎🗣️和零样本歌声转换🎶。无需任何训练，只需提供1~30秒的参考语音即可克隆声音。  

要查看演示列表和与之前语音转换模型的比较，请访问我们的 [演示页面](https://plachtaa.github.io/seed-vc/)🌐  

我们将继续改进模型质量并添加更多功能。

## 评估📊
### 零样本声音转换🎙🔁
我们对 Seed-VC 的语音转换能力进行了系列客观评估。  
为了便于复现，源音频是来自 LibriTTS-test-clean 的 100 个随机语句，参考音频是 12 个随机挑选的具有独特特征的自然声音。<br>  

源音频位于 `./examples/libritts-test-clean` <br>
参考音频位于 `./examples/reference` <br>

我们从说话人嵌入余弦相似度（SECS）、词错误率（WER）和字符错误率（CER）三个方面评估了转换结果，并将我们的结果与两个强大的开源基线模型，即 [OpenVoice](https://github.com/myshell-ai/OpenVoice) 和 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)，进行了比较。  
下表的结果显示，我们的 Seed-VC 模型在发音清晰度和说话人相似度上均显著优于基线模型。<br>

| 模型\指标         | SECS↑      | WER↓       | CER↓       | SIG↑     | BAK↑     | OVRL↑    |
|---------------|------------|------------|------------|----------|----------|----------|
| Ground Truth  | 1.0000     | 0.0802     | 0.0157     | ~        | ~        | ~        |
| OpenVoice     | 0.7547     | 0.1546     | 0.0473     | **3.56** | **4.02** | **3.27** |
| CosyVoice     | 0.8440     | 0.1898     | 0.0729     | 3.51     | **4.02** | 3.21     |
| Seed-VC（Ours） | **0.8676** | **0.1199** | **0.0292** | 3.42     | 3.97     | 3.11     |

我们也与非zero-shot的声线转换模型在特定角色上进行了比较（基于可以找到的公开模型）:

| Characters | Models\Metrics | SECS↑      | WER↓      | CER↓     | SIG↑     | BAK↑     | OVRL↑    |
|------------|----------------|------------|-----------|----------|----------|----------|----------|
| ~          | Ground Truth   | 1.0000     | 6.43      | 1.00     | ~        | ~        | ~        |
| 东海帝王       | So-VITS-4.0    | 0.8637     | 21.46     | 9.63     | 3.06     | 3.66     | 2.68     |
|            | Seed-VC(Ours)  | **0.8899** | **15.32** | **4.66** | **3.12** | **3.71** | **2.72** |
| 明前奶绿       | So-VITS-4.0    | 0.6850     | 48.43     | 32.50    | 3.34     | 3.51     | 2.82     |
|            | Seed-VC(Ours)  | **0.8072** | **7.26**  | **1.32** | **3.48** | **4.07** | **3.20** |
| 待兼诗歌剧      | So-VITS-4.0    | 0.8594     | 16.25     | 8.64     | **3.25** | 3.71     | 2.84     |
|            | Seed-VC(Ours)  | **0.8768** | **12.62** | **5.86** | 3.18     | **3.83** | **2.85** |

结果显示，即便我们的模型没有在特定说话人上进行微调或训练，在音色相似度和咬字清晰度上也全面优于在特定说话人数据集上专门训练的SoVITS模型。 
但是该项测试结果高度依赖于SoVITS模型质量。如果您认为此对比不公平或不够准确，欢迎提issue或PR。  
(东海帝王模型来自 [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))   
(待兼诗歌剧模型来自 [zomehwh/sovits-tannhauser](https://huggingface.co/spaces/zomehwh/sovits-tannhauser))  
(明前奶绿模型来自 [sparanoid/milky-green-sovits-4](https://huggingface.co/spaces/sparanoid/milky-green-sovits-4))  

*ASR 结果由 [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft) 模型计算*  
*说话人嵌入由 [resemblyzer](https://github.com/resemble-ai/Resemblyzer) 模型计算* <br>

你可以通过运行 `eval.py` 脚本来复现评估。  
```bash
python eval.py 
--source ./examples/libritts-test-clean
--target ./examples/reference
--output ./examples/eval/converted
--diffusion-steps 25
--length-adjust 1.0
--inference-cfg-rate 0.7
--xvector-extractor "resemblyzer"
--baseline ""  # 填入 openvoice 或 cosyvoice 来计算基线结果
--max-samples 100  # 要处理的最大源语句数
```
在此之前，如果你想运行基线评估，请确保已在 `../OpenVoice/` 和 `../CosyVoice/` 目录下正确安装了 openvoice 和 cosyvoice 仓库。

### 零样本歌声转换🎤🎶

我们在 [M4Singer](https://github.com/M4Singer/M4Singer) 数据集上进行了额外的歌声转换评估，使用了4个目标说话人，他们的音频数据可以在 [这里](https://huggingface.co/datasets/XzJosh/audiodataset) 获取。  
说话人相似性是通过将转换结果与各自角色数据集中所有可用样本的余弦相似性取平均来计算的。  
对于每个角色，随机选择一个语音作为零样本推理的提示。为了比较，我们为每个角色训练了相应的 [RVCv2-f0-48k](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 模型作为基线。  
每种歌手类型使用了100个随机语音作为源音频。

| Models\Metrics | F0CORR↑ | F0RMSE↓ | SECS↑      | CER↓      | SIG↑     | BAK↑     | OVRL↑    |
|----------------|---------|---------|------------|-----------|----------|----------|----------|
| RVCv2          | 0.9404  | 30.43   | 0.7264     | 28.46     | **3.41** | **4.05** | **3.12** |
| Seed-VC(Ours)  | 0.9375  | 33.35   | **0.7405** | **19.70** | 3.39     | 3.96     | 3.06     |

<details>
<summary>点击展开详细的评估结果</summary>

| 源音频类别 | 角色  | Models\Metrics | F0CORR↑ | F0RMSE↓ | SECS↑      | CER↓      | SIG↑ | BAK↑ | OVRL↑    |
|-------|--------|----------------|---------|---------|------------|-----------|------|------|----------|
| 女低音   | ~      | Ground Truth   | 1.0000  | 0.00    | ~          | 8.16      | ~    | ~    | ~        |
|       | 东雪莲（女） | RVCv2          | 0.9617  | 33.03   | **0.7352** | 24.70     | 3.36 | 4.07 | 3.07     |
|       |        | Seed-VC(Ours)  | 0.9658  | 31.64   | 0.7341     | **15.23** | 3.37 | 4.02 | 3.07     |
|       | 嘉然（女）  | RVCv2          | 0.9626  | 32.56   | 0.7212     | 19.67     | 3.45 | 4.08 | **3.17** |
|       |        | Seed-VC(Ours)  | 0.9648  | 31.94   | **0.7457** | **16.81** | 3.49 | 3.99 | 3.15     |
|       | 丁真（男）  | RVCv2          | 0.9013  | 26.72   | 0.7221     | 18.53     | 3.37 | 4.03 | 3.06     |
|       |        | Seed-VC(Ours)  | 0.9356  | 21.87   | **0.7513** | **15.63** | 3.44 | 3.94 | **3.09** |
|       | 科比（男）  | RVCv2          | 0.9215  | 23.90   | 0.7495     | 37.23     | 3.49 | 4.06 | **3.21** |
|       |        | Seed-VC(Ours)  | 0.9248  | 23.40   | **0.7602** | **26.98** | 3.43 | 4.02 | 3.13     |
| 男低音   | ~      | Ground Truth   | 1.0000  | 0.00    | ~          | 8.62      | ~    | ~    | ~        |
|       | 东雪莲    | RVCv2          | 0.9288  | 32.62   | **0.7148** | 24.88     | 3.45 | 4.10 | **3.18** |
|       |        | Seed-VC(Ours)  | 0.9383  | 31.57   | 0.6960     | **10.31** | 3.45 | 4.03 | 3.15     |
|       | 嘉然     | RVCv2          | 0.9403  | 30.00   | 0.7010     | 14.54     | 3.53 | 4.15 | **3.27** |
|       |        | Seed-VC(Ours)  | 0.9428  | 30.06   | **0.7299** | **9.66**  | 3.53 | 4.11 | 3.25     |
|       | 丁真     | RVCv2          | 0.9061  | 19.53   | 0.6922     | 25.99     | 3.36 | 4.09 | **3.08** |
|       |        | Seed-VC(Ours)  | 0.9169  | 18.15   | **0.7260** | **14.13** | 3.38 | 3.98 | 3.07     |
|       | 科比     | RVCv2          | 0.9302  | 16.37   | 0.7717     | 41.04     | 3.51 | 4.13 | **3.25** |
|       |        | Seed-VC(Ours)  | 0.9176  | 17.93   | **0.7798** | **24.23** | 3.42 | 4.08 | 3.17     |
| 女高音   | ~      | Ground Truth   | 1.0000  | 0.00    | ~          | 27.92     | ~    | ~    | ~        |
|       | 东雪莲    | RVCv2          | 0.9742  | 47.80   | 0.7104     | 38.70     | 3.14 | 3.85 | **2.83** |
|       |        | Seed-VC(Ours)  | 0.9521  | 64.00   | **0.7177** | **33.10** | 3.15 | 3.86 | 2.81     |
|       | 嘉然     | RVCv2          | 0.9754  | 46.59   | **0.7319** | 32.36     | 3.14 | 3.85 | **2.83** |
|       |        | Seed-VC(Ours)  | 0.9573  | 59.70   | 0.7317     | **30.57** | 3.11 | 3.78 | 2.74     |
|       | 丁真     | RVCv2          | 0.9543  | 31.45   | 0.6792     | 40.80     | 3.41 | 4.08 | **3.14** |
|       |        | Seed-VC(Ours)  | 0.9486  | 33.37   | **0.6979** | **34.45** | 3.41 | 3.97 | 3.10     |
|       | 科比     | RVCv2          | 0.9691  | 25.50   | 0.6276     | 61.59     | 3.43 | 4.04 | **3.15** |
|       |        | Seed-VC(Ours)  | 0.9496  | 32.76   | **0.6683** | **39.82** | 3.32 | 3.98 | 3.04     |
| 男高音   | ~      | Ground Truth   | 1.0000  | 0.00    | ~          | 5.94      | ~    | ~    | ~        |
|       | 东雪莲    | RVCv2          | 0.9333  | 42.09   | **0.7832** | 16.66     | 3.46 | 4.07 | **3.18** |
|       |        | Seed-VC(Ours)  | 0.9162  | 48.06   | 0.7697     | **8.48**  | 3.38 | 3.89 | 3.01     |
|       | 嘉然     | RVCv2          | 0.9467  | 36.65   | 0.7729     | 15.28     | 3.53 | 4.08 | **3.24** |
|       |        | Seed-VC(Ours)  | 0.9360  | 41.49   | **0.7920** | **8.55**  | 3.49 | 3.93 | 3.13     |
|       | 丁真     | RVCv2          | 0.9197  | 22.82   | 0.7591     | 12.92     | 3.40 | 4.02 | **3.09** |
|       |        | Seed-VC(Ours)  | 0.9247  | 22.77   | **0.7721** | **13.95** | 3.45 | 3.82 | 3.05     |
|       | 科比     | RVCv2          | 0.9415  | 19.33   | 0.7507     | 30.52     | 3.48 | 4.02 | **3.19** |
|       |        | Seed-VC(Ours)  | 0.9082  | 24.86   | **0.7764** | **13.35** | 3.39 | 3.93 | 3.07     |
</details>  

尽管Seed-VC没有在目标说话人上进行训练，并且只使用了一个随机语音作为提示，但它在说话人相似性（SECS）和可懂度（CER）方面仍然持续超越了在特定说话人的上百条语音上训练的RVCv2模型，这展示了Seed-VC卓越的语音克隆能力和鲁棒性。  

然而，观察到Seed-VC的音频质量（DNSMOS）略低于RVCv2。我们对此缺点非常重视，并将在未来优先尝试改善音频质量。  
如果您发现此比较不公平或不准确，欢迎提交PR或Issue。

*中文ASR结果由 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 计算*  
*说话人嵌入由 [resemblyzer](https://github.com/resemble-ai/Resemblyzer) 模型计算*  
*我们为男转女设置了+12半音的音高移位，女转男设置了-12半音的音高移位，否则为0音高移位*
## 安装 📥
建议在 Windows 或 Linux 上使用 Python 3.10：
```bash
pip install -r requirements.txt
```

## 使用方法🛠️
首次运行推理时，将自动下载最新模型的检查点。  

命令行推理：
```bash
python inference.py --source <源语音文件路径>
--target <参考语音文件路径>
--output <输出目录>
--diffusion-steps 25 # 建议歌声转换时使用50~100
--length-adjust 1.0
--inference-cfg-rate 0.7
--f0-condition False # 歌声转换时设置为 True
--auto-f0-adjust False # 设置为 True 可自动调整源音高到目标音高，歌声转换中通常不使用
--semi-tone-shift 0 # 歌声转换的半音移调
```
其中:
- `source` 待转换为参考声音的源语音文件路径
- `target` 声音参考的语音文件路径
- `output` 输出目录的路径
- `diffusion-steps` 使用的扩散步数，默认25，最佳质量建议使用50-100，最快推理使用4-10
- `length-adjust` 长度调整系数，默认1.0，<1.0加速语音，>1.0减慢语音
- `inference-cfg-rate` 对输出有细微影响，默认0.7
- `f0-condition` 是否根据源音频的音高调整输出音高，默认 False，歌声转换时设置为 True  
- `auto-f0-adjust` 是否自动将源音高调整到目标音高水平，默认 False，歌声转换中通常不使用
- `semi-tone-shift` 歌声转换中的半音移调，默认0  

Gradio 网页界面:
```bash
python app.py
```

实时变声界面:
```bash
python real-time-gui.py
```
强烈建议使用GPU进行实时声音转换任务。  
以下为部分在 NVIDIA RTX 3060 Laptop GPU 上的实验结果, 以及几组推荐的参数设置:

| Remarks                                 | Diffusion Steps | Inference CFG Rate | Max Prompt Length | Block Time (s) | Crossfade Length (s) | Extra context (left) (s) | Extra context (right) (s) | Latency (ms) | Quality | Inference Time per Chunk (ms) |
|-----------------------------------------|-----------------|--------------------|-------------------|----------------|----------------------|--------------------------|---------------------------|--------------|---------|-------------------------------| 
| 适合大多数声音                                 | 10              | 0.7                | 3.0               | 1.0s           | 0.04s                | 0.5s                     | 0.02s                     | 2070ms       | Medium  | 849ms                         |
| 音质会更好                                   | 20              | 0.7                | 3.0               | 2.0s           | 0.04s                | 0.5s                     | 0.02s                     | 4070ms       | High    | 1585ms                        |
| 音质差一些，但是对部分男性声音听不出差距                    | 5               | 0.7                | 3.0               | 0.6s           | 0.04s                | 0.5s                     | 0.02s                     | 1270ms       | Low     | 488ms                         |
| 将inference_cfg_rate调为0.0可提速，但不确定性能是否有下降 | 10              | 0.0                | 3.0               | 0.7s           | 0.04s                | 0.5s                     | 0.02s                     | 1470ms       | Medium  | 555ms                         |

你可以根据自己的硬件情况自由调整参数，只要Inference Time 小于 Block Time，声音转换即可正常运行.  
在PC运行其它GPU任务时，推理速度会有所下降 (例如：运行游戏或播放影片)  
总体来说，受制于推理时间，目前需要1~2秒延迟来避免质量下降（扩散模型推理慢的通病），我们会努力寻找更好的解决方案。  

*(GUI and audio chunking logic are modified from [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), thanks for their brilliant implementation!)*

然后在浏览器中打开 `http://localhost:7860/` 使用网页界面。
## TODO📝
- [x] 发布代码
- [x] 发布 v0.1 预训练模型： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-SeedVC-blue)](https://huggingface.co/Plachta/Seed-VC)
- [x] Hugging Face Space 演示： [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Plachta/Seed-VC)
- [x] HTML 演示页面（可能包含与其他 VC 模型的比较）： [Demo](https://plachtaa.github.io/seed-vc/)
- [x] 流式推理
- [ ] 降低延迟
- [ ] 实时变声Demo视频
- [x] 歌声转换
- [x] 提高源音频抗噪性
- [ ] 潜在的架构改进
    - [x] 类似U-ViT 的skip connection
    - [x] 将输入更改为 OpenAI Whisper
- [ ] 自定义数据训练代码
- [x] 歌声解码器更改为 NVIDIA 的 BigVGAN
- [x] 44k Hz 歌声转换模型
- [x] 歌声转换的客观指标评估以及与RVC/SoVITS模型的比较
- [ ] 提升音质
- [ ] 更多待添加

## 更新日志 🗒️
- 2024-10-27:
    - 更新了实时变声脚本
- 2024-10-25:
    - 添加了详尽的歌声转换评估结果以及与RVCv2模型的比较
- 2024-10-24:
    - 更新了44kHz歌声转换模型
- 2024-10-07:
    - 更新了 v0.3 预训练模型，将语音内容编码器更改为 OpenAI Whisper
    - 添加了 v0.3 预训练模型的客观指标评估结果
- 2024-09-22:
    - 将歌声转换模型的解码器更改为 BigVGAN，解决了大部分高音部分无法正确转换的问题
    - 在Web UI中支持对长输入音频的分段处理以及流式输出
- 2024-09-18:
    - 更新了用于歌声转换的模型
- 2024-09-14:
    - 更新了 v0.2 预训练模型，具有更小的尺寸和更少的扩散步骤即可达到相同质量，且增加了控制韵律保留的能力
    - 添加了命令行推理脚本
    - 添加了安装和使用说明
