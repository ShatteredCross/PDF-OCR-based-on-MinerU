"""
@license AGPL-3.0
Copyright (c) 2025 ShatteredCross. All rights reserved.
"""
from datetime import datetime
import os
import tempfile
import shutil
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# 定义输出时的颜色常量
YELLOW = '\033[93m'
GREEN = '\033[92m'
WHITE = '\033[0m'
RED = '\033[91m'

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print(f"{YELLOW}警告: 未安装pdf2image，PDF功能将不可用。请运行: pip install pdf2image")

def convert_pdf_to_images(pdf_path, output_dir=None, dpi=200):
    """
    将PDF转换为多张图片
    """
    if not PDF_SUPPORT:
        raise ImportError("pdf2image未安装，无法处理PDF文件")
    
    # 创建临时目录（如果未指定）
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"{YELLOW}正在将PDF转换为图片，保存到: {WHITE}{output_dir}")
    
    # 转换PDF为图片
    images = convert_from_path(pdf_path, dpi=dpi)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f"{YELLOW}已保存第 {i+1} 页: {WHITE}{image_path}")
    
    return image_paths, output_dir

def cleanup_temp_files(temp_dir):
    """
    清理临时文件
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"{YELLOW}已清理临时文件: {WHITE}{temp_dir}")

def process_single_image(image_path, client):
    """
    处理单张图片的OCR
    """
    print(f"{YELLOW}正在处理图片: {WHITE}{image_path}")
    image = Image.open(image_path)
    extracted_blocks = client.two_step_extract(image)
    return extracted_blocks

def save_ocr_results_as_formatted_md(all_extracted_blocks, original_path, multipage=False):
    """
    将OCR识别结果渲染为格式化的Markdown页面
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(original_path))[0]
    
    if multipage:
        filename = f"{original_name}_[OCR_Multipage]_{timestamp}.md"
    else:
        filename = f"{original_name}_[OCR]_{timestamp}.md"
    
    # 生成格式化的Markdown内容
    md_content = generate_formatted_markdown(all_extracted_blocks, original_name, multipage)
    
    # 确保输出目录存在
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # 构建完整的输出路径
    output_path = os.path.join(output_dir, filename)
    
    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"{GREEN}格式化Markdown已保存为: {output_path}")
    return output_path

def generate_formatted_markdown(all_extracted_blocks, original_name, multipage=False):
    """生成格式化的Markdown内容"""
    
    content = []
    
    if multipage:
        content.append(f"## OCR识别结果 - {original_name} (多页文档)\n")
    else:
        content.append(f"## OCR识别结果 - {original_name}\n")
        
    content.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    if multipage:
        content.append(f"*总页数: {len(all_extracted_blocks)}*\n")
    
    content.append("\n---\n\n")
    
    # 处理多页或单页内容
    if multipage:
        for page_num, page_blocks in enumerate(all_extracted_blocks, 1):
            content.append(f"### 第 {page_num} 页\n\n")
            content.extend(process_blocks(page_blocks))
            if page_num < len(all_extracted_blocks):  # 不是最后一页
                content.append("\n---\n\n")
    else:
        content.extend(process_blocks(all_extracted_blocks))
    
    # 添加数学公式支持说明
    content.append("\n---\n")
    content.append("*本文档包含数学公式，如需正确渲染请确保查看环境支持MathJax或KaTeX*")
    
    return "".join(content)

def process_blocks(blocks):
    """处理单个页面的块内容"""
    content_lines = []
    
    for i, block in enumerate(blocks):
        block_type = block.get('type', 'unknown')
        block_content = block.get('content')

        # 检查 block_content 是否为 None 或空
        if block_content is None or not block_content.strip():
            continue
            
        # 去除首尾空白
        block_content = block_content.strip()
            
        # 根据类型处理内容
        if block_type == 'equation':
            # 数学公式 - 直接使用LaTeX格式
            content_lines.append(block_content + "\n\n")
        elif block_type == 'footer':
            # 页脚 - 可以特殊处理或当作普通文本
            content_lines.append(f"*{block_content}*\n\n")
        elif block_type == 'header':
            # 页眉 - 可以特殊处理或当作普通文本
            content_lines.append(f"*{block_content}*\n\n")
        else:
            # 其他文本类型
            content_lines.append(block_content + "\n\n")
    
    return content_lines


def initialize_model_and_processor(model_path):
    """初始化模型和处理器"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=True
    )
    
    return model, processor


def main():
    # -----------------------------------------------------------------
    # 在这里输入【模型文件】存放于本地的绝对路径
    model_path = r"这里放模型文件夹在本地的绝对路径"
    # -----------------------------------------------------------------
    
    # -----------------------------------------------------------------
    # 在这里输入【待识别文件】存放于本地的绝对路径（支持图片和PDF）
    input_path = r"这里放需要识别的图片或PDF的绝对路径"
    # -----------------------------------------------------------------
    
    # 初始化模型
    model, processor = initialize_model_and_processor(model_path)
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor
    )
    
    temp_dir = None
    try:
        # 检查文件类型
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == '.pdf':
            if not PDF_SUPPORT:
                print(f"{RED}错误: PDF支持未启用，请安装pdf2image")
                return
            
            print(f"{GREEN}检测到PDF文件，开始转换...")
            # 转换PDF为图片
            image_paths, temp_dir = convert_pdf_to_images(input_path)
            
            # 处理每一页
            all_blocks = []
            for image_path in image_paths:
                blocks = process_single_image(image_path, client)
                all_blocks.append(blocks)
            
            # 保存为多页Markdown
            output_path = save_ocr_results_as_formatted_md(all_blocks, input_path, multipage=True)
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"{GREEN}检测到图片文件，开始处理...")
            # 处理单张图片
            blocks = process_single_image(input_path, client)
            output_path = save_ocr_results_as_formatted_md(blocks, input_path, multipage=False)
            
        else:
            print(f"{RED}错误: 不支持的文件格式 {file_ext}")
            return
            
        print(f"{GREEN}OCR处理完成! 结果保存在: {output_path}")
        
    finally:
        # 清理临时文件
        if temp_dir:
            cleanup_temp_files(temp_dir)


if __name__ == "__main__":
    main()