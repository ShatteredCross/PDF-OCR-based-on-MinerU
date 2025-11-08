# PDF OCR based on MinerU

<p align="center">
  <a href="README_zh.md">中文</a> •
  <a href="README.md">English</a>
</p>


A PDF and image document recognition tool based on the MinerU2.5-1.2B OCR large language model.

## Project Description

This project implements OCR (Optical Character Recognition) functionality for PDF documents and image files using the open-source MinerU2.5-1.2B visual language model from OpenDataLab. The tool can extract text content from scanned documents, images, or PDF files and convert them into structured Markdown format files.

### Core Features

- **Multi-format Support**: Handles `PDF` documents and common image formats (`JPG`, `JPEG`, `PNG`, `BMP`)
- **Intelligent Recognition**: Based on the advanced MinerU2.5 model, capable of recognizing various document elements including complex papers, mathematical formulas, headers, footers, tables, and images
- **Structured Output**: Generates formatted Markdown files that preserve the original document's layout structure
- **Batch Processing**: Automatically processes multi-page PDFs, performing page-by-page recognition and merging outputs

### Technical Features

- 🌐 **Cross-platform Compatibility**: Supports Windows, Linux, and macOS (via Docker)
- 📦 **One-click Deployment**: Provides Docker images for out-of-the-box use
- 🖥️ **User-friendly Interface**: Modern GUI based on Gradio with Chinese/English language switching
- 📐 **Precise Formatting**: Supports LaTeX format output for complex mathematical formulas
- 🧩 **Intelligent Parsing**: Automatically processes text blocks in different document locations

### Performance

- In practical testing, the tool demonstrates good recognition accuracy for printed text, high-quality scanned documents, and mathematical formulas.
- For specific performance data, please refer to the [MinerU team's test results on HuggingFace](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B).
- Recognition time and effectiveness may be affected by original document quality, font clarity, and layout complexity.

## Quick Start

### Environment Requirements

- Python 3.8 or higher
- CUDA-supported GPU (recommended) or CPU-only
- Pre-installed PyTorch

This project has been tested with `Python 3.13`, `cu126`, and `torch2.8.0`.

### Dependency Installation

```bash
git clone https://github.com/ShatteredCross/PDF-OCR-based-on-MinerU.git && cd PDF-OCR-based-on-MinerU

pip install -r requirements.txt
```

Additionally, the `pdf2image` library depends on `poppler-utils`. Ubuntu / Debian users can install it via:
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

Windows users can install via Conda package manager:
```powershell
conda install -c conda-forge poppler
```

### Model Download

Choose one of the following two download methods.

#### Using Hugging Face Hub

The required dependency libraries for downloading model files via Hugging Face Hub are already included in `requirements.txt`. If you haven't completed the **Dependency Installation** step, please return to the previous section.

If you have good network connectivity, you can skip this step. If network conditions prevent access to Hugging Face, configure environment variables to use a mirror site as the download endpoint:

- **Linux / macOS**
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- **Windows PowerShell**
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  ```

Then, create and execute a Python script file for downloading (this step cannot be skipped):
```python
from huggingface_hub import snapshot_download
# Download model to local cache directory
local_dir = snapshot_download(repo_id="opendatalab/MinerU2.5-2509-1.2B")
print("Model downloaded to:", local_dir)
```

Note the output `local_dir`, as you'll need to provide the absolute path to the model directory when running our main program `web_demo.py`.

#### Using Git Clone

We do not recommend this download method, as `git clone` will download the entire repository's Git history and structure. Even with `--depth=1` and `--filter=blob:none`, it may still download many unnecessary files.

```bash
git clone --depth=1 --filter=blob:none https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/
```

### Usage Methods

#### 1. Using Docker

> Recommended for users who want to use the tool in an isolated environment or prefer out-of-the-box solutions.

Our Docker image has been tested on Windows systems and supports GPU inference. Built with the following configuration (refer to the `Dockerfile` for details):
- Docker Engine v28.3.3
- pytorch:2.8.0-cuda12.6-cudnn9-runtime

You can choose to **build from source** or **directly pull our published image from Docker Hub** (recommended).

- Build from source:
    ```bash
    docker build -t pdf-ocr-based-on-mineru:v1.0.0 .
    ```
- Pull the latest published image from Docker Hub **(recommended, one-click deployment)**:
    ```bash
    docker pull shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
    ```

To run the image, use the following command:
```bash
docker run -d --gpus all -p 8100:8100 -v /absolute/path/to/model/on/host:/app/checkpoints --name pdf-ocr shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
```

⚠️ **Note**: Users need to manually replace `/absolute/path/to/model/on/host` with the **actual** model storage path. For example, if your model is stored at `D:/MinerU2.5-2509-1.2B`, the command should be:

```bash
docker run -d --gpus all -p 8100:8100 -v D:/MinerU2.5-2509-1.2B:/app/checkpoints --name pdf-ocr shatteredcross/pdf-ocr-based-on-mineru:v1.0.0
```

After completing the volume mount, Docker users should enter `/app/checkpoints` as the model path in the browser interface.

#### 2. Using `web_demo.py`

> Recommended for users unfamiliar with code or who prefer ready-to-use solutions.

First, run the `web_demo.py` script:
```bash
python web_demo.py
```

Generally, the program will automatically run and open the following URL in your browser, which is the main service port. If it doesn't open automatically after a while, try manually entering the URL in your browser:
```url
http://localhost:8100
```

If successfully opened, users will see the following interface:
![](/fig_and_test_example/fig_of_web_demo_en.png)

> ⚠️ **Note**: Users can switch language display via the `中文/English` button in the top-right corner. Currently supported languages are Chinese and English.

Usage instructions are provided at the bottom of the webpage, which we reiterate here:
1. **Set Model Path**: Enter the absolute path to the `MinerU2.5-1.2B` model folder on your local machine (for Docker users, enter `/app/checkpoints`)
    > ⚠️ **Note**: The `local_dir` output during the model download stage is the absolute path where the model was downloaded locally
2. **Click Load Model**: Wait for model loading to complete (status bar shows success)
3. **Upload File**: Supports `PDF`, `JPG`, `JPEG`, `PNG`, `BMP` formats
4. **Start Recognition**: Click the Start OCR Recognition button and wait for processing to complete
5. **View Results**: Check recognition results and download Markdown files on the right side

#### 3. Using `basic_demo.py`

> Recommended for users familiar with code or who prefer full control over the process.

1. **Manually Configure Model Path**: Modify the `model_path` variable in the code to point to the local model directory
    ```python
    # -----------------------------------------------------------------
    # Enter the absolute path to the model directory here
    model_path = r"Enter the absolute path to the model directory here"
    # -----------------------------------------------------------------
    ```
2. **Manually Specify Input File**: Set the `input_path` variable to point to the PDF or image file to be recognized
    ```python
    # -----------------------------------------------------------------
    # Enter the absolute path to the file to be recognized here (supports images and PDF)
    input_path = r"Enter the absolute path to the image or PDF file to be recognized here"
    # -----------------------------------------------------------------
    ```
3. **Run Recognition Program**: Execute the main program `basic_demo.py` or `basic_demo_en.py`. For differences between them, refer to the "Code Description" section below.

### Output Results

The program will generate Markdown files containing OCR recognition results in the `output/` folder at the project root directory. File name formats are:
- Single file: `original_filename_[OCR]_timestamp.md`
- Multi-page PDF: `original_filename_[OCR_Multipage]_timestamp.md`

## Code Description

- `web_demo.py`: This is the project's demonstration script, containing core functionality and a modern frontend interface implemented with Gradio. Suitable for users unfamiliar with code or who prefer ready-to-use solutions.
    > In the browser interface, users can switch language display via the `中文/English` button in the top-right corner. Currently supported languages are Chinese and English.

- `basic_demo.py`: This is the project's basic script, containing core functionality but without a frontend interface. Suitable for users familiar with code or who prefer full control over the process. Can be run directly in an IDE to view console output.

- `basic_demo_en.py`: This is the English translation version of `basic_demo.py`, with identical functionality to `basic_demo.py`.
    > For the convenience of English-speaking users, the program's interactive output messages and code comments in this version have been translated into English.
    > 
    > Please note that the translation was performed by AI and may contain technical terminology inaccuracies, grammatical errors, or semantic deviations. If in doubt, please refer to the original Chinese version.

## Demonstration of Recognition Results

> Please note that OCR recognition may yield slightly different results for the same image across multiple recognition attempts due to factors such as model characteristics, runtime environment, or subtle variations in input images. We do not guarantee absolute consistency in recognition results and assume no responsibility for any direct or indirect losses arising from the use of this project.
> 
> OCR recognition results are for reference only.

Using the file `Lagrangian of the Standard Model (Low Resolution).pdf` from the `fig_and_test_example/` folder for OCR testing, we obtained the following results:
![](/fig_and_test_example/fig_Recognition_Results_Demonstration.png)

## Acknowledgments

This project is developed based on the open-source MinerU2.5-1.2B OCR large language model from OpenDataLab. We extend special thanks to the OpenDataLab team for their outstanding contributions and open-source spirit in the field of document intelligence recognition.

- **MinerU Project Address**: https://github.com/opendatalab/MinerU/

Thanks to all developers who contribute code and models to the open-source community. Your efforts drive the popularization and development of artificial intelligence technology.

---

*Note: The recognition effectiveness of this tool may be affected by original document quality, model training data, and specific usage environment. We recommend conducting sufficient testing and verification before practical application.*