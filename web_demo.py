"""
@license AGPL-3.0
Copyright (c) 2025 ShatteredCross. All rights reserved.
"""
from datetime import datetime
import os
import tempfile
import shutil
from pathlib import Path
import gradio as gr
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# 全局变量，用于保存模型和客户端
global_model = None
global_processor = None
global_client = None

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# 多语言文本定义
TEXTS = {
    "zh": {
        "title": "PDF OCR based on MinerU2.5-1.2B",
        "subtitle": "基于 MinerU2.5-1.2B OCR 大模型的 PDF 和图片文档识别工具",
        "model_path_label": "模型路径",
        "model_path_placeholder": "请输入模型文件夹的绝对路径...（如为Docker,输入 /app/checkpoints ）",
        "load_model_btn": "加载模型",
        "file_input_label": "上传文件",
        "process_btn": "开始OCR识别",
        "status_output_label": "处理状态",
        "result_output_label": "识别结果 (Markdown格式)",
        "file_output_label": "下载结果文件",
        "instructions_title": "使用说明",
        "instructions": [
            "1. **设置模型路径**: 输入 `MinerU2.5-1.2B` 模型文件夹的绝对路径（如为Docker,输入 /app/checkpoints ）",
            "2. **点击加载模型**: 等待模型加载完成（状态栏显示成功）",
            "3. **上传文件**: 支持 PDF、JPG、JPEG、PNG、BMP 格式",
            "4. **开始识别**: 点击开始OCR识别按钮，等待处理完成",
            "5. **查看结果**: 在右侧查看识别结果和下载Markdown文件"
        ],
        "supported_formats_title": "支持的格式",
        "supported_formats": [
            "PDF 文档（多页自动处理）",
            "图片文件: JPG, JPEG, PNG, BMP"
        ],
        "notes_title": "注意事项",
        "notes": [
            "处理大型PDF文件可能需要较长时间",
            "确保模型路径正确且包含完整的模型文件"
        ],
        "language_btn": "English",
        # 新增的状态和错误信息
        "model_not_loaded": "❌ 请先加载模型！点击上方的'加载模型'按钮完成模型初始化后再进行OCR识别。",
        "no_file_uploaded": "❌ 请先上传要识别的文件！",
        "pdf_not_supported": "❌ PDF支持未启用，请安装pdf2image: pip install pdf2image",
        "unsupported_format": "❌ 不支持的文件格式 {file_ext}",
        "file_detected": "📄 检测到文件: {filename}",
        "pdf_detected": "🔄 检测到PDF文件，开始转换...",
        "pdf_converted": "✅ PDF转换完成，共 {page_count} 页",
        "image_detected": "🖼️ 检测到图片文件，开始处理...",
        "page_processed": "✅ 第 {page_num} 页处理完成",
        "image_processed": "✅ 图片处理完成",
        "ocr_completed": "✅ OCR处理完成! 结果保存在: {filename}",
        "processing_error": "❌ 处理过程中发生错误: {error}",
        "model_loading": "正在加载模型...",
        "processor_loading": "正在加载处理器...",
        "client_initializing": "正在初始化客户端...",
        "model_loaded": "模型加载完成",
        "model_load_success": "✅ 模型加载成功！",
        "model_path_not_exist": "❌ 错误: 模型路径不存在",
        "model_load_failed": "❌ 模型加载失败: {error}",
        "pdf_converting": "正在转换PDF为图片...",
        "processing_page": "正在处理第 {page_num} 页...",
        "processing_image": "正在处理图片...",
        "generating_markdown": "正在生成Markdown文件...",
        "processing_complete": "处理完成"
    },
    "en": {
        "title": "PDF OCR based on MinerU2.5-1.2B",
        "subtitle": "PDF and Image OCR Tool based on MinerU2.5-1.2B Model",
        "model_path_label": "Model Path",
        "model_path_placeholder": "Please enter the absolute path to the model directory...(For Docker, input /app/checkpoints)",
        "load_model_btn": "Load Model",
        "file_input_label": "Upload File",
        "process_btn": "Start OCR Recognition",
        "status_output_label": "Processing Status",
        "result_output_label": "Recognition Result (Markdown Format)",
        "file_output_label": "Download Result File",
        "instructions_title": "Usage Instructions",
        "instructions": [
            "1. **Set Model Path**: Enter the absolute path to the `MinerU2.5-1.2B` model directory (For Docker, input /app/checkpoints)",
            "2. **Click Load Model**: Wait for model loading to complete (status bar shows success)",
            "3. **Upload File**: Supports PDF, JPG, JPEG, PNG, BMP formats",
            "4. **Start Recognition**: Click the Start OCR Recognition button and wait for processing to complete",
            "5. **View Results**: Check the recognition results and download Markdown file on the right"
        ],
        "supported_formats_title": "Supported Formats",
        "supported_formats": [
            "PDF documents (multi-page automatic processing)",
            "Image files: JPG, JPEG, PNG, BMP"
        ],
        "notes_title": "Notes",
        "notes": [
            "Processing large PDF files may take a long time",
            "Ensure the model path is correct and contains complete model files"
        ],
        "language_btn": "中文",
        # 状态和错误信息
        "model_not_loaded": "❌ Please load the model first! Click the 'Load Model' button above to complete model initialization before OCR recognition.",
        "no_file_uploaded": "❌ Please upload a file to recognize first!",
        "pdf_not_supported": "❌ PDF support is not enabled, please install pdf2image: pip install pdf2image",
        "unsupported_format": "❌ Unsupported file format {file_ext}",
        "file_detected": "📄 File detected: {filename}",
        "pdf_detected": "🔄 PDF file detected, starting conversion...",
        "pdf_converted": "✅ PDF conversion completed, total {page_count} pages",
        "image_detected": "🖼️ Image file detected, starting processing...",
        "page_processed": "✅ Page {page_num} processed",
        "image_processed": "✅ Image processing completed",
        "ocr_completed": "✅ OCR processing completed! Results saved to: {filename}",
        "processing_error": "❌ Error occurred during processing: {error}",
        "model_loading": "Loading model...",
        "processor_loading": "Loading processor...",
        "client_initializing": "Initializing client...",
        "model_loaded": "Model loading completed",
        "model_load_success": "✅ Model loaded successfully!",
        "model_path_not_exist": "❌ Error: Model path does not exist",
        "model_load_failed": "❌ Model loading failed: {error}",
        "pdf_converting": "Converting PDF to images...",
        "processing_page": "Processing page {page_num}...",
        "processing_image": "Processing image...",
        "generating_markdown": "Generating Markdown file...",
        "processing_complete": "Processing completed"
    }
}

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
    
    # 转换PDF为图片
    images = convert_from_path(pdf_path, dpi=dpi)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths, output_dir

def process_single_image(image_path, client):
    """
    处理单张图片的OCR
    """
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
    
    return output_path, md_content  # 这里返回完整路径

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

def initialize_model(model_path, current_lang, progress=gr.Progress()):
    """
    初始化模型和处理器
    """
    global global_model, global_processor, global_client
    
    try:
        progress(0.1, desc=TEXTS[current_lang]["model_loading"])
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            return None, TEXTS[current_lang]["model_path_not_exist"]
        
        global_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype="auto",
            device_map="auto"
        )

        progress(0.6, desc=TEXTS[current_lang]["processor_loading"])
        
        global_processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True
        )
        
        progress(0.9, desc=TEXTS[current_lang]["client_initializing"])
        
        global_client = MinerUClient(
            backend="transformers",
            model=global_model,
            processor=global_processor
        )
        
        progress(1.0, desc=TEXTS[current_lang]["model_loaded"])
        return global_client, TEXTS[current_lang]["model_load_success"]
        
    except Exception as e:
        return None, TEXTS[current_lang]["model_load_failed"].format(error=str(e))

def process_file(input_file, current_lang, progress=gr.Progress()):
    """
    处理上传的文件
    """
    global global_model, global_processor, global_client
    
    if global_model is None or global_processor is None or global_client is None:
        return gr.update(), gr.update(), TEXTS[current_lang]["model_not_loaded"]
    
    if input_file is None:
        return gr.update(), gr.update(), TEXTS[current_lang]["no_file_uploaded"]
    
    temp_dir = None
    status_messages = []
    
    # 定义进度区间
    progress_ranges = {
        'pdf_conversion': (0.0, 0.2),      # 20%
        'page_processing': (0.2, 0.9),     # 70% 
        'markdown_generation': (0.9, 0.95), # 5%
        'completion': (0.95, 1.0)           # 5%
    }
    
    try:
        # 检查文件类型
        file_ext = os.path.splitext(input_file.name)[1].lower()
        status_messages.append(TEXTS[current_lang]["file_detected"].format(filename=os.path.basename(input_file.name)))
        
        if file_ext == '.pdf':
            if not PDF_SUPPORT:
                return gr.update(), gr.update(), TEXTS[current_lang]["pdf_not_supported"]
            
            status_messages.append(TEXTS[current_lang]["pdf_detected"])
            progress(progress_ranges['pdf_conversion'][1], desc=TEXTS[current_lang]["pdf_converting"])
            
            # 转换PDF为图片
            image_paths, temp_dir = convert_pdf_to_images(input_file.name)
            status_messages.append(TEXTS[current_lang]["pdf_converted"].format(page_count=len(image_paths)))
            
            # 处理每一页 - 使用进度区间计算
            all_blocks = []
            page_start, page_end = progress_ranges['page_processing']
            for i, image_path in enumerate(image_paths):
                # 计算当前页面处理的进度
                current_progress = page_start + (i / len(image_paths)) * (page_end - page_start)
                progress(current_progress, desc=TEXTS[current_lang]["processing_page"].format(page_num=i+1))
                blocks = process_single_image(image_path, global_client)
                all_blocks.append(blocks)
                status_messages.append(TEXTS[current_lang]["page_processed"].format(page_num=i+1))
            
            progress(progress_ranges['markdown_generation'][1], desc=TEXTS[current_lang]["generating_markdown"])
            # 保存为多页Markdown
            md_file, md_content = save_ocr_results_as_formatted_md(all_blocks, input_file.name, multipage=True)
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            status_messages.append(TEXTS[current_lang]["image_detected"])
            # 单张图片处理直接使用页面处理的结束点
            progress(progress_ranges['page_processing'][1], desc=TEXTS[current_lang]["processing_image"])
            
            # 处理单张图片
            blocks = process_single_image(input_file.name, global_client)
            status_messages.append(TEXTS[current_lang]["image_processed"])
            
            progress(progress_ranges['markdown_generation'][1], desc=TEXTS[current_lang]["generating_markdown"])
            md_file, md_content = save_ocr_results_as_formatted_md(blocks, input_file.name, multipage=False)
            
        else:
            return gr.update(), gr.update(), TEXTS[current_lang]["unsupported_format"].format(file_ext=file_ext)
        
        progress(progress_ranges['completion'][1], desc=TEXTS[current_lang]["processing_complete"])
        status_messages.append(TEXTS[current_lang]["ocr_completed"].format(filename=md_file))
        
        # 返回结果
        status_text = "\n".join(status_messages)
        return md_content, md_file, status_text
        
    except Exception as e:
        status_messages.append(TEXTS[current_lang]["processing_error"].format(error=str(e)))
        status_text = "\n".join(status_messages)
        return gr.update(), gr.update(), status_text
        
    finally:
        # 清理临时文件
        if temp_dir:
            cleanup_temp_files(temp_dir)

def cleanup_temp_files(temp_dir):
    """
    清理临时文件
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def create_gradio_interface():
    """
    创建Gradio界面
    """
    with gr.Blocks(title="PDF OCR based on MinerU2.5-1.2B", theme=gr.themes.Soft()) as demo:
        # 语言状态
        current_lang = gr.State(value="zh")
        
        # 标题行
        with gr.Column():
            title_md = gr.Markdown("# PDF OCR based on MinerU2.5-1.2B")
            with gr.Row():
                with gr.Column(scale=5):
                    subtitle_md = gr.Markdown("基于 MinerU2.5-1.2B OCR 大模型的 PDF 和图片文档识别工具")
                with gr.Column(scale=1):
                    language_btn = gr.Button("English", size="sm")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 模型路径输入
                model_path = gr.Textbox(
                    label="模型路径",
                    placeholder="请输入模型文件夹的绝对路径...（如为Docker,输入 /app/checkpoints ）",
                    lines=2,        # 显示行数
                    max_lines=3,    # 最大行数，输入过长时自动滚动
                )
                
                # 模型加载按钮
                load_model_btn = gr.Button("加载模型", variant="primary")
                
                # 文件上传
                file_input = gr.File(
                    label="上传文件",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp"],
                    file_count="single"  # 明确指定单文件
                )
                
                # 处理按钮
                process_btn = gr.Button("开始OCR识别", variant="primary")
            
            with gr.Column(scale=3):
                # 状态显示
                status_output = gr.Textbox(
                    label="处理状态",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
                
                # 结果显示
                result_output = gr.Textbox(
                    label="识别结果 (Markdown格式)",
                    lines=20,
                    max_lines=25,
                    show_copy_button=True
                )
                
                # 文件下载
                file_output = gr.File(
                    label="下载结果文件",
                    file_types=[".md"]
                )
        
        # 说明区域
        with gr.Row(equal_height=True):
            gr.Column(scale=1, min_width=0)
            with gr.Column(scale=3):
                gr.Markdown("---")
                instructions_title = gr.Markdown("### 使用说明")
                instructions_content = gr.Markdown("""
                1. **设置模型路径**: 输入 `MinerU2.5-1.2B` 模型文件夹的绝对路径（如为Docker,输入 /app/checkpoints ）
                2. **点击加载模型**: 等待模型加载完成（状态栏显示成功）
                3. **上传文件**: 支持 PDF、JPG、JPEG、PNG、BMP 格式
                4. **开始识别**: 点击开始OCR识别按钮，等待处理完成
                5. **查看结果**: 在右侧查看识别结果和下载Markdown文件
                """)
                
                supported_formats_title = gr.Markdown("### 支持的格式")
                supported_formats_content = gr.Markdown("""
                - PDF 文档（多页自动处理）
                - 图片文件: JPG, JPEG, PNG, BMP
                """)
                
                notes_title = gr.Markdown("### 注意事项")
                notes_content = gr.Markdown("""
                - 处理大型PDF文件可能需要较长时间
                - 确保模型路径正确且包含完整的模型文件
                """)
            gr.Column(scale=1, min_width=0)
        
        # 语言切换函数
        def switch_language(lang):
            new_lang = "en" if lang == "zh" else "zh"
            texts = TEXTS[new_lang]
            
            # 生成说明内容
            instructions_text = "\n".join(texts["instructions"])
            supported_formats_text = "\n".join([f"- {item}" for item in texts["supported_formats"]])
            notes_text = "\n".join([f"- {item}" for item in texts["notes"]])
            
            return [
                gr.update(value=f"# {texts['title']}"),  # title_md
                gr.update(value=texts['subtitle']),     # subtitle_md
                gr.update(label=texts['model_path_label'], placeholder=texts['model_path_placeholder']),  # model_path
                gr.update(value=texts['load_model_btn']), # load_model_btn
                gr.update(label=texts['file_input_label']), # file_input
                gr.update(value=texts['process_btn']),    # process_btn
                gr.update(label=texts['status_output_label']),  # status_output
                gr.update(label=texts['result_output_label']),  # result_output
                gr.update(label=texts['file_output_label']), # file_output
                gr.update(value=f"### {texts['instructions_title']}"),  # instructions_title
                gr.update(value=instructions_text),     # instructions_content
                gr.update(value=f"### {texts['supported_formats_title']}"),  # supported_formats_title
                gr.update(value=supported_formats_text), # supported_formats_content
                gr.update(value=f"### {texts['notes_title']}"),  # notes_title
                gr.update(value=notes_text),            # notes_content
                gr.update(value=texts['language_btn']),   # language_btn
                new_lang                                        # current_lang
            ]
        
        # 事件处理
        load_model_btn.click(
            fn=initialize_model,
            inputs=[model_path, current_lang],
            outputs=[gr.Number(visible=False), status_output]
        )

        process_btn.click(
            fn=process_file,
            inputs=[file_input, current_lang],
            outputs=[result_output, file_output, status_output]
        )
        
        # 语言切换事件
        language_btn.click(
            fn=switch_language,
            inputs=[current_lang],
            outputs=[
                title_md, subtitle_md, model_path, load_model_btn, file_input,
                process_btn, status_output, result_output, file_output,
                instructions_title, instructions_content, supported_formats_title,
                supported_formats_content, notes_title, notes_content, language_btn,
                current_lang
            ]
        )
    
    return demo

if __name__ == "__main__":
    # 启动Gradio界面
    demo = create_gradio_interface()
    demo.launch(
        max_file_size="1000mb",         # 限制上传文件大小
        server_name="0.0.0.0",          # 允许外部访问
        server_port=8100,               # 端口号
        share=False,                    # 不生成公共链接
        inbrowser=True                  # 自动在浏览器中打开
    )