# PDF OCR based on MinerU

<p align="center">
  <a href="README_zh.md">中文</a> •
  <a href="README.md">English</a>
</p>

基于 MinerU2.5-1.2B OCR 大模型的 PDF 和图片文档识别工具。

## 项目描述

本项目基于 OpenDataLab 开源的 MinerU2.5-1.2B 视觉语言模型，实现了对 PDF 文档和图片文件的 OCR（光学字符识别）功能。该工具能够将扫描文档、图片或 PDF 文件中的文本内容提取出来，并转换为结构化的 Markdown 格式文件。

### 核心功能

- **多格式支持**：支持处理 `PDF` 文档以及常见的图片格式（`JPG`、`JPEG`、`PNG`、`BMP`）
- **智能识别**：基于先进的 MinerU2.5 模型，能够识别复杂论文、数学公式、页眉页脚、表格和图片等多种文档元素
- **结构化输出**：生成格式化的 Markdown 文件，保持原始文档的排版结构
- **批量处理**：自动处理多页 PDF，逐页识别后合并输出

### 技术特点

- 🌐 **跨平台兼容**：支持 Windows、Linux 和 macOS（通过 Docker）
- 📦 **一键部署**：提供 Docker 镜像，开箱即用
- 🖥️ **友好界面**：基于 Gradio 的现代化 GUI，支持中英文切换
- 📐 **精准排版**：支持复杂数学公式的 LaTeX 格式输出
- 🧩 **智能解析**：自动处理文档中不同位置的文本块

### 性能表现

- 在实际测试中，该工具对于印刷体文字、高清扫描文档和数学公式具有较好的识别准确率。
- 如果需要具体的数据，请参考 [MinerU 团队在 HuggingFace 发布页的测试结果](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)。
- 识别时间和识别效果会受到原始文档质量、字体清晰度和版面复杂度的影响。

## 快速开始

### 环境要求

- Python 3.8 或更高版本
- 支持 CUDA 的 GPU（推荐）或仅使用 CPU
- 需要预先安装好 PyTorch

本项目在 `Python 3.13`、`cu126`、`torch2.8.0` 下通过测试。

### 依赖安装

```bash
git clone https://github.com/ShatteredCross/PDF-OCR-based-on-MinerU.git && cd PDF-OCR-based-on-MinerU

pip install -r requirements.txt
```

此外，`pdf2image` 库依赖 `poppler-utils`，Ubuntu / Debian 用户可通过如下方式安装：
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

Windows 用户可以通过包管理器 Conda 安装：
```powershell
conda install -c conda-forge poppler
```

### 模型下载

以下提到的两种下载方式只需选择一种完成下载。

#### 使用 Hugging Face Hub 下载

使用 Hugging Face Hub 下载模型文件时所需的依赖库已在 `requirements.txt` 中包含，如果没有进行 **依赖安装** 步骤，请回到上面执行。

如果网络情况较好，这一步可以跳过；如果网络情况不足以支持访问 Hugging Face，需要在命令行里配置环境变量，让 `huggingface_hub` 使用镜像站作为下载端点：
- **Linux / macOS**
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- **Windows PowerShell**
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  ```

然后，创建一个实现下载功能的 `.py` Python 脚本文件并执行它（这一步不能跳过）
```python
from huggingface_hub import snapshot_download
# 下载模型到本地缓存目录
local_dir = snapshot_download(repo_id="opendatalab/MinerU2.5-2509-1.2B")
print("模型已下载到:", local_dir)
```

可以记录一下输出的 `local_dir`，在运行我们项目的主程序 `web_demo.py` 时，需要提供模型所在的绝对路径。

#### 使用 Git Clone 下载

我们不推荐使用这种方式下载，因为 `git clone` 会把整个仓库的 Git 历史和结构拉下来，即使加了 `--depth=1` 和 `--filter=blob:none`，仍然可能下载到很多不必要的文件。

```bash
git clone --depth=1 --filter=blob:none https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/
```

### 使用方法
#### 1. 通过 Docker 使用

> 对于希望在隔离的环境中使用或希望开箱即用的用户，我们推荐使用这种方式。

我们的 Docker 镜像目前在 Windows 系统上通过测试，可以使用 GPU 推理。基于如下配置编译，具体可以参考 `Dockerfile` 文件：
- Docker Engine v28.3.3
- pytorch:2.8.0-cuda12.6-cudnn9-runtime


您可以选择从 **源代码编译** 或者 **直接从 Docker Hub 获取我们发布的镜像** （推荐）。

- 从源代码编译
    ```bash
    docker build -t pdf-ocr-based-on-mineru:v1.0.0 .
    ```
- 直接从 Docker Hub 获取我们最新发布的镜像 **（推荐，一键部署）**
    ```bash
    docker pull shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
    ```

要运行镜像，请参考如下的代码：
```bash
docker run -d --gpus all -p 8100:8100 -v 模型在宿主机中的绝对路径:/app/checkpoints  --name pdf-ocr shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
```

⚠️ **注意**，用户需要手动替换 `模型在宿主机中的绝对路径` 为**真实的**模型存放路径。例如，假设你的模型放在 `D:/MinerU2.5-2509-1.2B` 下，那么运行代码就应该是

```bash
docker run -d --gpus all -p 8100:8100 -v D:/MinerU2.5-2509-1.2B:/app/checkpoints --name pdf-ocr shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
```

到这里我们已经完成了卷挂载，因此，当 Docker 用户在浏览器界面中被要求输入模型路径时，请输入 `/app/checkpoints`。

#### 2. 通过 `web_demo.py` 使用

> 对于不熟悉代码或希望开箱即用的用户，我们推荐使用这种方式。

首先运行 `web_demo.py` 脚本：
```bash
python web_demo.py
```

一般而言，程序会自动运行并在浏览器中打开如下的 `URL`，这也是程序提供服务的主要端口。如果长时间未能自动打开，请尝试手动在浏览器中输入访问如下的 `URL`。
```url
http://localhost:8100
```

若成功打开，用户会看到如下的界面：
![](/fig_and_test_example/fig_of_web_demo_zh.png)

> ⚠️ **注意**：用户可以通过右上角的 `中文/English` 按钮切换语言显示，目前支持的语言为中文和英语

在网页的最下方有使用说明，我们在这里再次给出：
1. **设置模型路径**: 输入 `MinerU2.5-1.2B` 模型文件夹在本地的绝对路径（如为 Docker 用户，输入 `/app/checkpoints` ）
    > ⚠️ **注意**：在模型下载阶段，脚本输出的 `local_dir` 就是模型下载到本地的绝对路径
2. **点击加载模型**: 等待模型加载完成（状态栏显示成功）
3. **上传文件**: 支持 `PDF`、`JPG`、`JPEG`、`PNG`、`BMP` 格式
4. **开始识别**: 点击开始OCR识别按钮，等待处理完成
5. **查看结果**: 在右侧查看识别结果和下载Markdown文件


#### 3. 通过 `basic_demo.py` 使用

> 对于熟悉代码或追求全流程可控的用户，我们推荐使用这种方式。

1. **手动配置模型路径**：在代码中修改 `model_path` 变量，指向本地模型目录
    
    ```python
    # -----------------------------------------------------------------
    # 在这里输入【模型文件】存放于本地的绝对路径
    model_path = r"这里放模型文件夹在本地的绝对路径"
    # -----------------------------------------------------------------
    ```
2. **手动指定输入文件**：设置 `input_path` 变量，指向待识别的 PDF 或图片文件
    ```python
    # -----------------------------------------------------------------
    # 在这里输入【待识别文件】存放于本地的绝对路径（支持图片和PDF）
    input_path = r"这里放需要识别的图片或PDF的绝对路径"
    # -----------------------------------------------------------------
    ```
3. **运行识别程序**：执行主程序 `basic_demo.py`，或者 `basic_demo_en.py`，关于二者的区别，可以在下文的“代码说明”中找到。

### 输出结果

程序将在项目根目录的 `output/` 文件夹中生成包含 OCR 识别结果的 Markdown 文件，文件名格式为：
- 单文件：`原文件名_[OCR]_时间戳.md`
- 多页 PDF：`原文件名_[OCR_Multipage]_时间戳.md`


## 代码说明
- `web_demo.py`：这是项目的演示脚本，包含核心功能和借助 Gradio 实现的现代化前端界面，适合不熟悉代码或希望开箱即用的用户
    > 在浏览器界面，用户可以通过右上角的 `中文/English` 按钮切换语言显示，目前支持的语言为中文和英语

- `basic_demo.py`：这是项目的基础脚本，包含核心功能，但没有前端显示，适合熟悉代码或追求全流程可控的用户，可以在 IDE 中直接运行，查看控制台输出

- `basic_demo_en.py`：这是 `basic_demo.py` 的英文翻译版本，功能与 `basic_demo.py` 一致
    > 为了方便英语使用者，在该版本中，程序在交互时的输出信息和代码注释都被翻译成了英文。
    > 
    > 请注意，翻译由 AI 完成，可能存在技术术语不准确、语法错误或语义偏差。如有疑问，请以原始中文版本为准。

## 识别效果展示
> 请注意，OCR 识别功能可能由于模型本身、运行环境或输入图像的细微差异等原因，导致对同一张图片的多次识别结果不完全一致。我们不保证识别结果的绝对一致性，也不对因使用本项目而产生的任何直接或间接损失承担责任。
> 
> OCR识别结果仅供参考。

使用 `fig_and_test_example\` 文件夹下的 `Lagrangian of the Standard Model ( Low Resolution ).pdf` 文件进行 OCR 测试，我们得到如下的结果：
![](/fig_and_test_example/fig_Recognition_Results_Demonstration.png)

## 致谢

本项目基于 OpenDataLab 开源的 MinerU2.5-1.2B OCR 大模型开发，特此感谢 OpenDataLab 团队在文档智能识别领域做出的杰出贡献和开源精神。

- **MinerU 项目地址**：https://github.com/opendatalab/MinerU/

感谢所有为开源社区贡献代码和模型的开发者们，正是你们的努力推动了人工智能技术的普及和发展。

---

*注意：本工具识别效果会受到原始文档质量、模型训练数据和具体使用环境的影响，建议在实际应用前进行充分的测试和验证。*