from datetime import datetime
import os
import tempfile
import shutil
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# Define color constants for output
YELLOW = '\033[93m'
GREEN = '\033[92m'
WHITE = '\033[0m'
RED = '\033[91m'

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print(f"{YELLOW}Warning: pdf2image is not installed, PDF functionality will be unavailable. Please run: pip install pdf2image")

def convert_pdf_to_images(pdf_path, output_dir=None, dpi=200):
    """
    Convert PDF to multiple images
    """
    if not PDF_SUPPORT:
        raise ImportError("pdf2image is not installed, cannot process PDF files")
    
    # Create temporary directory (if not specified)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"{YELLOW}Converting PDF to images, saving to: {WHITE}{output_dir}")
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
        print(f"{YELLOW}Saved page {i+1}: {WHITE}{image_path}")
    
    return image_paths, output_dir

def cleanup_temp_files(temp_dir):
    """
    Clean up temporary files
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"{YELLOW}Cleaned up temporary files: {WHITE}{temp_dir}")

def process_single_image(image_path, client):
    """
    Process single image for OCR
    """
    print(f"{YELLOW}Processing image: {WHITE}{image_path}")
    image = Image.open(image_path)
    extracted_blocks = client.two_step_extract(image)
    return extracted_blocks

def save_ocr_results_as_formatted_md(all_extracted_blocks, original_path, multipage=False):
    """
    Render OCR results as formatted Markdown page
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(original_path))[0]
    
    if multipage:
        filename = f"{original_name}_[OCR_Multipage]_{timestamp}.md"
    else:
        filename = f"{original_name}_[OCR]_{timestamp}.md"
    
    # Generate formatted Markdown content
    md_content = generate_formatted_markdown(all_extracted_blocks, original_name, multipage)
    
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # Build complete output path
    output_path = os.path.join(output_dir, filename)
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"{GREEN}Formatted Markdown saved as: {output_path}")
    return output_path

def generate_formatted_markdown(all_extracted_blocks, original_name, multipage=False):
    """Generate formatted Markdown content"""
    
    content = []
    
    if multipage:
        content.append(f"## OCR Results - {original_name} (Multi-page Document)\n")
    else:
        content.append(f"## OCR Results - {original_name}\n")
        
    content.append(f"*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    if multipage:
        content.append(f"*Total pages: {len(all_extracted_blocks)}*\n")
    
    content.append("\n---\n\n")
    
    # Process multi-page or single-page content
    if multipage:
        for page_num, page_blocks in enumerate(all_extracted_blocks, 1):
            content.append(f"### Page {page_num}\n\n")
            content.extend(process_blocks(page_blocks))
            if page_num < len(all_extracted_blocks):  # Not the last page
                content.append("\n---\n\n")
    else:
        content.extend(process_blocks(all_extracted_blocks))
    
    # Add math formula support note
    content.append("\n---\n")
    content.append("*This document contains mathematical formulas. Please ensure your viewing environment supports MathJax or KaTeX for proper rendering*")
    
    return "".join(content)

def process_blocks(blocks):
    """Process block content for a single page"""
    content_lines = []
    
    for i, block in enumerate(blocks):
        block_type = block.get('type', 'unknown')
        block_content = block.get('content')

        # Check if block_content is None or empty
        if block_content is None or not block_content.strip():
            continue
            
        # Remove leading/trailing whitespace
        block_content = block_content.strip()
            
        # Process content based on type
        if block_type == 'equation':
            # Mathematical formulas - use LaTeX format directly
            content_lines.append(block_content + "\n\n")
        elif block_type == 'footer':
            # Footer - can be specially treated or as regular text
            content_lines.append(f"*{block_content}*\n\n")
        elif block_type == 'header':
            # Header - can be specially treated or as regular text
            content_lines.append(f"*{block_content}*\n\n")
        else:
            # Other text types
            content_lines.append(block_content + "\n\n")
    
    return content_lines


def initialize_model_and_processor(model_path):
    """Initialize model and processor"""
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
    # Enter the absolute path to the model directory here
    model_path = r"Enter the absolute path to the model directory here"
    # -----------------------------------------------------------------
    
    # -----------------------------------------------------------------
    # Enter the absolute path to the file to be recognized here (supports images and PDF)
    input_path = r"Enter the absolute path to the image or PDF file to be recognized here"
    # -----------------------------------------------------------------
    
    # Initialize model
    model, processor = initialize_model_and_processor(model_path)
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor
    )
    
    temp_dir = None
    try:
        # Check file type
        file_ext = os.path.splitext(input_path)[1].lower()
        
        if file_ext == '.pdf':
            if not PDF_SUPPORT:
                print(f"{RED}Error: PDF support is not enabled, please install pdf2image")
                return
            
            print(f"{GREEN}PDF file detected, starting conversion...")
            # Convert PDF to images
            image_paths, temp_dir = convert_pdf_to_images(input_path)
            
            # Process each page
            all_blocks = []
            for image_path in image_paths:
                blocks = process_single_image(image_path, client)
                all_blocks.append(blocks)
            
            # Save as multi-page Markdown
            output_path = save_ocr_results_as_formatted_md(all_blocks, input_path, multipage=True)
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"{GREEN}Image file detected, starting processing...")
            # Process single image
            blocks = process_single_image(input_path, client)
            output_path = save_ocr_results_as_formatted_md(blocks, input_path, multipage=False)
            
        else:
            print(f"{RED}Error: Unsupported file format {file_ext}")
            return
            
        print(f"{GREEN}OCR processing completed! Results saved to: {output_path}")
        
    finally:
        # Clean up temporary files
        if temp_dir:
            cleanup_temp_files(temp_dir)


if __name__ == "__main__":
    main()