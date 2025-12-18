from flask import Flask, render_template, request, redirect, send_file, jsonify, session
from flask_socketio import SocketIO, emit, join_room
import io
import zipfile
import json
import warnings
import os
import sys
import uuid
import threading
import re
from collections import OrderedDict

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detect_bubbles import detect_bubbles
from process_bubble import process_bubble_auto
from translator.translator import MangaTranslator
from translator.context_memory import ContextMemory
from add_text import add_text
from manga_ocr import MangaOcr
from ocr.chrome_lens_ocr import ChromeLensOCR
from PIL import Image
import numpy as np
import base64
import cv2


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "manga_translator_secret_key_2024")

# Initialize SocketIO
def get_async_mode():
    if getattr(sys, 'frozen', False):
        return 'threading'
    try:
        import eventlet
        return 'eventlet'
    except ImportError:
        pass
    try:
        import gevent
        return 'gevent'
    except ImportError:
        pass
    return 'threading'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=get_async_mode())

# Session storage for manga data
manga_sessions = OrderedDict()
MAX_SESSIONS = 50

def cleanup_old_sessions():
    while len(manga_sessions) > MAX_SESSIONS:
        manga_sessions.popitem(last=False)

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"

def log(msg):
    if VERBOSE_LOG:
        print(msg)

MODEL_PATH = "model/model.pt"
DEFAULT_SPLIT_HEIGHT_RATIO = 2.0


def natural_sort_key(s):
    """Sort strings with numbers naturally: ch1, ch2, ch10 instead of ch1, ch10, ch2"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def split_long_image(image: np.ndarray, max_height_ratio: float = DEFAULT_SPLIT_HEIGHT_RATIO) -> list:
    height, width = image.shape[:2]
    max_height = int(width * max_height_ratio)
    
    if height <= max_height:
        return [image]
    
    chunks = []
    current_y = 0
    
    while current_y < height:
        chunk_end = min(current_y + max_height, height)
        chunk = image[current_y:chunk_end, :].copy()
        chunks.append(chunk)
        current_y = chunk_end
    
    return chunks


@app.route("/")
def home():
    return render_template("index.html")


def get_font_path(font_name: str) -> str:
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


def process_chapter_images(images_data, manga_translator, mocr, selected_font, translator_type, 
                          session_id, chapter_id, enable_black_bubble=True, use_context_memory=True):
    """
    Process all images in a chapter with progress reporting.
    Uses progressive context building for consistent translations.
    """
    
    total_images = len(images_data)
    all_pages_data = {}
    all_bubble_images = []
    bubble_mapping = []
    
    # Phase 1: Detection
    emit_chapter_progress(session_id, chapter_id, 'processing', 10, 'Phát hiện speech bubbles...')
    
    for idx, img_data in enumerate(images_data):
        image = img_data['image']
        name = img_data['name']
        
        results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
        if not results:
            all_pages_data[name] = {'image': image, 'bubbles': [], 'texts': []}
            continue
        
        bubble_data = []
        for bubble_idx, result in enumerate(results):
            if len(result) >= 7:
                x1, y1, x2, y2, score, class_id, is_dark = result[:7]
            else:
                x1, y1, x2, y2, score, class_id = result[:6]
                is_dark = 0
            
            detected_image = image[int(y1):int(y2), int(x1):int(x2)]
            all_bubble_images.append(Image.fromarray(detected_image.copy()))
            bubble_mapping.append((name, bubble_idx))
            
            processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(detected_image, force_dark=(is_dark == 1))
            
            bubble_data.append({
                'detected_image': processed_image,
                'contour': cont,
                'coords': (int(x1), int(y1), int(x2), int(y2)),
                'is_dark': bubble_is_dark,
                'fill_color': detected_color
            })
        
        all_pages_data[name] = {'image': image, 'bubbles': bubble_data, 'texts': []}
        
        progress = 10 + int((idx + 1) / total_images * 20)
        emit_chapter_progress(session_id, chapter_id, 'processing', progress, f'Phát hiện: {idx+1}/{total_images}')
    
    # Phase 2: OCR
    emit_chapter_progress(session_id, chapter_id, 'processing', 35, 'Nhận dạng văn bản (OCR)...')
    
    if all_bubble_images:
        use_batch_ocr = hasattr(mocr, 'process_batch')
        if use_batch_ocr:
            all_texts = mocr.process_batch(all_bubble_images)
        else:
            all_texts = [mocr(img) for img in all_bubble_images]
        
        for (page_name, bubble_idx), text in zip(bubble_mapping, all_texts):
            all_pages_data[page_name]['texts'].append(text)
    
    emit_chapter_progress(session_id, chapter_id, 'processing', 50, 'Đang dịch văn bản...')
    
    # Phase 3: Translation with Progressive Context
    pages_texts = {name: data['texts'] for name, data in all_pages_data.items() if data['texts']}
    all_translations = {}
    
    if pages_texts:
        translator = None
        if translator_type == "copilot" and hasattr(manga_translator, '_local_llm_translator'):
            translator = manga_translator._local_llm_translator
        elif translator_type == "gemini" and hasattr(manga_translator, '_gemini_translator'):
            translator = manga_translator._gemini_translator
        elif translator_type == "openrouter" and hasattr(manga_translator, '_openrouter_translator'):
            translator = manga_translator._openrouter_translator
        
        if translator:
            # Create shared context memory for the entire chapter
            context_memory = ContextMemory() if use_context_memory else None
            
            # Split into batches of 5 pages for progressive context building
            page_names = list(pages_texts.keys())
            batch_size = 5
            
            for batch_start in range(0, len(page_names), batch_size):
                batch_end = min(batch_start + batch_size, len(page_names))
                batch_names = page_names[batch_start:batch_end]
                batch_texts = {name: pages_texts[name] for name in batch_names}
                
                try:
                    # Translate batch with accumulated context
                    translated = translator.translate_pages_batch(
                        batch_texts, 
                        manga_translator.source, 
                        manga_translator.target, 
                        context_memory=context_memory
                    )
                    all_translations.update(translated)
                    
                    # Update context with this batch's translations for next batch
                    if context_memory:
                        context_memory.update_from_batch(batch_texts, translated)
                        print(f"📝 Context updated: {context_memory.get_stats()}")
                    
                except Exception as e:
                    print(f"Batch translation failed: {e}")
                    # Fallback: keep original texts
                    for name in batch_names:
                        all_translations[name] = pages_texts[name]
                
                # Update progress
                progress = 50 + int((batch_end / len(page_names)) * 25)
                emit_chapter_progress(session_id, chapter_id, 'processing', progress, 
                                     f'Dịch: {batch_end}/{len(page_names)} trang')
    
    emit_chapter_progress(session_id, chapter_id, 'processing', 80, 'Render text vào ảnh...')
    
    # Phase 4: Render
    processed_results = []
    font_path = get_font_path(selected_font)
    
    for name, data in all_pages_data.items():
        image = data['image']
        bubbles = data['bubbles']
        translated_texts = all_translations.get(name, data['texts'])
        
        for bubble, text in zip(bubbles, translated_texts):
            x1, y1, x2, y2 = bubble['coords']
            bubble_region = image[y1:y2, x1:x2]
            text_color = (255, 255, 255) if bubble.get('is_dark', False) else (0, 0, 0)
            add_text(bubble_region, text, font_path, bubble['contour'], text_color)
        
        processed_results.append({'image': image, 'name': name})
    
    emit_chapter_progress(session_id, chapter_id, 'completed', 100, 'Hoàn tất!')
    
    return processed_results


def emit_chapter_progress(session_id, chapter_id, status, progress, message):
    """Emit chapter processing progress via Socket.IO"""
    chapter_name = ""
    if session_id in manga_sessions:
        for ch in manga_sessions[session_id].get('chapters', []):
            if ch['id'] == chapter_id:
                chapter_name = ch['name']
                ch['status'] = status
                ch['progress'] = progress
                break
    
    socketio.emit('chapter_progress', {
        'session_id': session_id,
        'chapter_id': chapter_id,
        'chapter_name': chapter_name,
        'status': status,
        'progress': progress,
        'message': message
    })


def process_chapter_background(session_id, chapter_id, config):
    """Background thread to process a chapter"""
    try:
        if session_id not in manga_sessions:
            return
        
        session_data = manga_sessions[session_id]
        chapter = None
        for ch in session_data['chapters']:
            if ch['id'] == chapter_id:
                chapter = ch
                break
        
        if not chapter or 'raw_images' not in chapter:
            return
        
        images_data = chapter['raw_images']
        
        # Initialize translator
        manga_translator = MangaTranslator(source=config['source_lang'], target=config['target_lang'])
        
        if config['translator_type'] == 'gemini':
            from translator.gemini_translator import GeminiTranslator
            manga_translator._gemini_translator = GeminiTranslator(
                api_key=config.get('gemini_api_key'),
                custom_prompt=config.get('custom_prompt')
            )
        elif config['translator_type'] == 'openrouter':
            from translator.openrouter_translator import OpenRouterTranslator
            manga_translator._openrouter_translator = OpenRouterTranslator(
                api_key=config.get('openrouter_api_key'),
                provider=config.get('openrouter_provider', 'claude'),
                custom_prompt=config.get('custom_prompt')
            )
        elif config['translator_type'] == 'copilot':
            from translator.local_llm_translator import LocalLLMTranslator
            manga_translator._local_llm_translator = LocalLLMTranslator(
                server_url=config.get('copilot_server', 'http://localhost:8080'),
                model=config.get('copilot_model', 'gpt-4o'),
                custom_prompt=config.get('custom_prompt')
            )
        
        # Initialize OCR
        if config.get('ocr_engine') == 'manga-ocr':
            mocr = MangaOcr()
        else:
            mocr = ChromeLensOCR()
        
        # Process images
        results = process_chapter_images(
            images_data, manga_translator, mocr,
            config['font'], config['translator_type'],
            session_id, chapter_id,
            enable_black_bubble=config.get('enable_black_bubble', True),
            use_context_memory=config.get('use_context_memory', True)
        )
        
        # Encode and store results
        encoded_images = []
        for result in results:
            _, buffer = cv2.imencode(".jpg", result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
            encoded_images.append({'name': result['name'], 'data': encoded})
        
        # Update session - remove raw_images to free memory
        chapter['images'] = encoded_images
        chapter['status'] = 'completed'
        del chapter['raw_images']
        
        # Start next pending chapter
        start_next_chapter(session_id, config)
        
    except Exception as e:
        print(f"Error processing chapter {chapter_id}: {e}")
        import traceback
        traceback.print_exc()
        emit_chapter_progress(session_id, chapter_id, 'error', 0, str(e))


def start_next_chapter(session_id, config):
    """Start processing the next pending chapter"""
    if session_id not in manga_sessions:
        return
    
    session_data = manga_sessions[session_id]
    
    for chapter in session_data['chapters']:
        if chapter['status'] == 'pending' and 'raw_images' in chapter:
            chapter['status'] = 'processing'
            
            thread = threading.Thread(
                target=process_chapter_background,
                args=(session_id, chapter['id'], config)
            )
            thread.daemon = True
            thread.start()
            break


@socketio.on('join_session')
def on_join_session(data):
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)


@app.route("/translate", methods=["POST"])
def upload_file():
    # Get form data - parse translator selection
    selected_translator_raw = request.form.get("selected_translator", "Gemini")
    
    # Handle emoji prefixes in translator names
    translator_map = {
        "gemini": "gemini", "🌟 gemini": "gemini",
        "openrouter": "openrouter", "🚀 openrouter": "openrouter",
        "local llm": "copilot", "💻 local llm": "copilot",
        "google": "google", "🔤 google": "google",
        "nllb": "nllb", "📦 nllb": "nllb",
        "baidu": "baidu", "🇨🇳 baidu": "baidu",
        "bing": "bing", "🔍 bing": "bing",
        "opus-mt model": "hf"
    }
    selected_translator = translator_map.get(selected_translator_raw.lower(), "gemini")
    
    # Get settings
    copilot_server = request.form.get("copilot_server", "http://localhost:8080")
    copilot_model = request.form.get("copilot_model_input", "gpt-4o")
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    openrouter_provider = request.form.get("openrouter_provider", "free")
    
    # OpenRouter key is hardcoded - no user input needed
    openrouter_api_key = "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"
    
    use_context_memory = request.form.get("context_memory") == "on"
    enable_black_bubble = request.form.get("detect_black_bubbles") == "on"
    split_long_images = request.form.get("split_long_images") == "on"
    
    selected_font_raw = request.form.get("selected_font", "Animeace")
    selected_font = selected_font_raw.lower()
    if selected_font == "animeace":
        selected_font = "animeace_"
    elif selected_font_raw.startswith("Yuki-"):
        selected_font = selected_font_raw
    
    selected_ocr = request.form.get("selected_ocr", "chrome-lens").lower()
    
    source_lang_map = {"japanese (manga)": "ja", "chinese (manhua)": "zh", "korean (manhwa)": "ko", "english (comic)": "en"}
    source_lang = source_lang_map.get(request.form.get("selected_source_lang", "Japanese (Manga)").lower(), "ja")
    
    target_lang_map = {"english": "en", "vietnamese": "vi", "chinese": "zh", "korean": "ko", "thai": "th",
                       "indonesian": "id", "french": "fr", "german": "de", "spanish": "es", "russian": "ru"}
    target_lang = target_lang_map.get(request.form.get("selected_language", "Vietnamese").lower(), "vi")
    
    style_map = {"default": "", "casual (thân mật)": "casual", "formal (trang trọng)": "formal",
                 "keep honorifics (-san, senpai...)": "keep_honorifics", "web novel style": "web_novel",
                 "action (ngắn gọn)": "action", "literal (sát nghĩa)": "literal"}
    style = style_map.get(request.form.get("selected_style", "Default").lower(), "")
    custom_prompt = request.form.get("custom_prompt", "").strip() or style
    
    upload_mode = request.form.get("upload_mode", "files")
    
    # Config for processing
    config = {
        'translator_type': selected_translator,
        'gemini_api_key': gemini_api_key,
        'openrouter_api_key': openrouter_api_key,
        'openrouter_provider': openrouter_provider,
        'copilot_server': copilot_server,
        'copilot_model': copilot_model,
        'custom_prompt': custom_prompt,
        'font': selected_font,
        'ocr_engine': selected_ocr,
        'source_lang': source_lang,
        'target_lang': target_lang,
        'enable_black_bubble': enable_black_bubble,
        'use_context_memory': use_context_memory,
        'split_long_images': split_long_images
    }
    
    # Handle different upload modes
    if upload_mode == 'files':
        files = request.files.getlist("files")
        if not files or files[0].filename == '':
            return redirect("/")
        return process_simple_upload(files, config)
    
    elif upload_mode == 'folder':
        # Single folder = 1 chapter
        files = request.files.getlist("folder")
        if not files:
            return redirect("/")
        return process_single_chapter_upload(files, config)
    
    elif upload_mode == 'multi-folder':
        # Multiple folders = multiple chapters
        files = request.files.getlist("multi-folder")
        if not files:
            return redirect("/")
        return process_multi_chapter_upload(files, config)
    
    return redirect("/")


def process_simple_upload(files, config):
    """Process simple file upload (original behavior) - returns translate.html"""
    from app_helpers import process_images_with_batch
    
    manga_translator = MangaTranslator(source=config['source_lang'], target=config['target_lang'])
    
    if config['translator_type'] == "gemini" and config.get('gemini_api_key'):
        manga_translator._gemini_api_key = config['gemini_api_key']
        manga_translator._gemini_custom_prompt = config.get('custom_prompt')
    
    if config['translator_type'] == "copilot":
        manga_translator._copilot_server = config['copilot_server']
        manga_translator._copilot_model = config['copilot_model']
        manga_translator._copilot_custom_prompt = config.get('custom_prompt')
    
    if config.get('ocr_engine') == 'manga-ocr':
        mocr = MangaOcr()
    else:
        mocr = ChromeLensOCR()
    
    # Read all images
    all_images = []
    for file in files:
        if file and file.filename:
            try:
                file_bytes = np.frombuffer(file.stream.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    name = os.path.splitext(os.path.basename(file.filename))[0]
                    all_images.append({'image': image, 'name': name})
            except Exception as e:
                print(f"Error reading {file.filename}: {e}")
    
    if not all_images:
        return redirect("/")
    
    # Sort images naturally
    all_images.sort(key=lambda x: natural_sort_key(x['name']))
    
    # Initialize translators
    if config['translator_type'] == 'gemini':
        from translator.gemini_translator import GeminiTranslator
        manga_translator._gemini_translator = GeminiTranslator(
            api_key=config.get('gemini_api_key'),
            custom_prompt=config.get('custom_prompt')
        )
    elif config['translator_type'] == 'openrouter':
        from translator.openrouter_translator import OpenRouterTranslator
        manga_translator._openrouter_translator = OpenRouterTranslator(
            api_key=config.get('openrouter_api_key'),
            provider=config.get('openrouter_provider', 'claude'),
            custom_prompt=config.get('custom_prompt')
        )
    elif config['translator_type'] == 'copilot':
        from translator.local_llm_translator import LocalLLMTranslator
        manga_translator._local_llm_translator = LocalLLMTranslator(
            server_url=config['copilot_server'],
            model=config['copilot_model'],
            custom_prompt=config.get('custom_prompt')
        )
    
    # Process
    # Note: batch_size nhỏ hơn để tránh gửi prompt quá lớn lên OpenRouter (tránh 413 Request Entity Too Large)
    processed_results = process_images_with_batch(
        all_images, manga_translator, mocr, config['font'],
        config['translator_type'], batch_size=4,
        use_context_memory=config['use_context_memory'],
        enable_black_bubble=config['enable_black_bubble'],
        socketio=socketio
    )
    
    # Encode results
    processed_images = []
    for result in processed_results:
        image = result['image']
        base_name = result['name']
        
        if config['split_long_images']:
            chunks = split_long_image(image)
        else:
            chunks = [image]
        
        for i, chunk in enumerate(chunks):
            _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
            encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
            chunk_name = f"{base_name}_part{i+1}" if len(chunks) > 1 else base_name
            processed_images.append({"name": chunk_name, "data": encoded})
    
    if not processed_images:
        return redirect("/")
    
    return render_template("translate.html", images=processed_images)


def process_single_chapter_upload(files, config):
    """Process single folder upload as 1 chapter - goes to reader"""
    
    # Filter image files and read them
    image_files = []
    folder_name = "Chapter 1"
    
    for file in files:
        if file and file.filename:
            # Check if it's an image
            if not re.search(r'\.(jpg|jpeg|png|webp)$', file.filename, re.I):
                continue
            
            # Get folder name from path
            path_parts = file.filename.replace('\\', '/').split('/')
            if len(path_parts) >= 2:
                folder_name = path_parts[0]  # Root folder name
            
            try:
                file_bytes = np.frombuffer(file.stream.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    name = os.path.splitext(os.path.basename(file.filename))[0]
                    image_files.append({
                        'image': image,
                        'name': name,
                        'filename': file.filename
                    })
            except Exception as e:
                print(f"Error reading {file.filename}: {e}")
    
    if not image_files:
        return redirect("/")
    
    # Sort images naturally
    image_files.sort(key=lambda x: natural_sort_key(x['name']))
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    cleanup_old_sessions()
    
    # Create single chapter
    chapter = {
        'id': 1,
        'name': folder_name,
        'status': 'processing',
        'progress': 0,
        'images': [],
        'raw_images': image_files  # Will be removed after processing
    }
    
    manga_sessions[session_id] = {
        'title': folder_name,
        'chapters': [chapter],
        'config': config
    }
    
    # Process immediately
    manga_translator = MangaTranslator(source=config['source_lang'], target=config['target_lang'])
    
    if config['translator_type'] == 'gemini':
        from translator.gemini_translator import GeminiTranslator
        manga_translator._gemini_translator = GeminiTranslator(
            api_key=config.get('gemini_api_key'),
            custom_prompt=config.get('custom_prompt')
        )
    elif config['translator_type'] == 'openrouter':
        from translator.openrouter_translator import OpenRouterTranslator
        manga_translator._openrouter_translator = OpenRouterTranslator(
            api_key=config.get('openrouter_api_key'),
            provider=config.get('openrouter_provider', 'claude'),
            custom_prompt=config.get('custom_prompt')
        )
    elif config['translator_type'] == 'copilot':
        from translator.local_llm_translator import LocalLLMTranslator
        manga_translator._local_llm_translator = LocalLLMTranslator(
            server_url=config['copilot_server'],
            model=config['copilot_model'],
            custom_prompt=config.get('custom_prompt')
        )
    
    if config.get('ocr_engine') == 'manga-ocr':
        mocr = MangaOcr()
    else:
        mocr = ChromeLensOCR()
    
    # Process chapter
    results = process_chapter_images(
        image_files, manga_translator, mocr,
        config['font'], config['translator_type'],
        session_id, 1,
        enable_black_bubble=config.get('enable_black_bubble', True),
        use_context_memory=config.get('use_context_memory', True)
    )
    
    # Encode results
    encoded_images = []
    for result in results:
        _, buffer = cv2.imencode(".jpg", result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        encoded_images.append({'name': result['name'], 'data': encoded})
    
    chapter['images'] = encoded_images
    chapter['status'] = 'completed'
    del chapter['raw_images']  # Free memory
    
    return redirect(f"/reader/{session_id}/1")


def process_multi_chapter_upload(files, config):
    """Process multiple folders as multiple chapters"""
    
    # Group files by subfolder
    folders = {}
    root_folder_name = "Manga"
    
    for file in files:
        if file and file.filename:
            if not re.search(r'\.(jpg|jpeg|png|webp)$', file.filename, re.I):
                continue
            
            path_parts = file.filename.replace('\\', '/').split('/')
            
            # Get root folder and subfolder
            if len(path_parts) >= 3:
                root_folder_name = path_parts[0]
                subfolder = path_parts[1]
            elif len(path_parts) >= 2:
                root_folder_name = path_parts[0]
                subfolder = path_parts[0]  # Same as root if no subfolder
            else:
                subfolder = "Chapter 1"
            
            if subfolder not in folders:
                folders[subfolder] = []
            
            try:
                file_bytes = np.frombuffer(file.stream.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    name = os.path.splitext(os.path.basename(file.filename))[0]
                    folders[subfolder].append({
                        'image': image,
                        'name': name,
                        'filename': file.filename
                    })
            except Exception as e:
                print(f"Error reading {file.filename}: {e}")
    
    if not folders:
        return redirect("/")
    
    # Sort folders naturally
    sorted_folder_names = sorted(folders.keys(), key=natural_sort_key)
    
    # Sort images within each folder
    for folder_name in sorted_folder_names:
        folders[folder_name].sort(key=lambda x: natural_sort_key(x['name']))
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    cleanup_old_sessions()
    
    # Create chapters
    chapters = []
    for idx, folder_name in enumerate(sorted_folder_names):
        chapter = {
            'id': idx + 1,
            'name': f"Ch.{idx + 1}: {folder_name}",
            'status': 'pending',
            'progress': 0,
            'images': [],
            'raw_images': folders[folder_name]
        }
        chapters.append(chapter)
    
    manga_sessions[session_id] = {
        'title': root_folder_name,
        'chapters': chapters,
        'config': config
    }
    
    # Process first chapter immediately
    if chapters:
        first_chapter = chapters[0]
        first_chapter['status'] = 'processing'
        
        manga_translator = MangaTranslator(source=config['source_lang'], target=config['target_lang'])
        
        if config['translator_type'] == 'gemini':
            from translator.gemini_translator import GeminiTranslator
            manga_translator._gemini_translator = GeminiTranslator(
                api_key=config.get('gemini_api_key'),
                custom_prompt=config.get('custom_prompt')
            )
        elif config['translator_type'] == 'openrouter':
            from translator.openrouter_translator import OpenRouterTranslator
            manga_translator._openrouter_translator = OpenRouterTranslator(
                api_key=config.get('openrouter_api_key'),
                provider=config.get('openrouter_provider', 'claude'),
                custom_prompt=config.get('custom_prompt')
            )
        elif config['translator_type'] == 'copilot':
            from translator.local_llm_translator import LocalLLMTranslator
            manga_translator._local_llm_translator = LocalLLMTranslator(
                server_url=config['copilot_server'],
                model=config['copilot_model'],
                custom_prompt=config.get('custom_prompt')
            )
        
        if config.get('ocr_engine') == 'manga-ocr':
            mocr = MangaOcr()
        else:
            mocr = ChromeLensOCR()
        
        results = process_chapter_images(
            first_chapter['raw_images'], manga_translator, mocr,
            config['font'], config['translator_type'],
            session_id, first_chapter['id'],
            enable_black_bubble=config.get('enable_black_bubble', True),
            use_context_memory=config.get('use_context_memory', True)
        )
        
        encoded_images = []
        for result in results:
            _, buffer = cv2.imencode(".jpg", result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
            encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
            encoded_images.append({'name': result['name'], 'data': encoded})
        
        first_chapter['images'] = encoded_images
        first_chapter['status'] = 'completed'
        del first_chapter['raw_images']
        
        # Start background processing for remaining chapters
        if len(chapters) > 1:
            thread = threading.Thread(
                target=process_remaining_chapters_background,
                args=(session_id, config)
            )
            thread.daemon = True
            thread.start()
    
    return redirect(f"/reader/{session_id}/1")


def process_remaining_chapters_background(session_id, config):
    """Process remaining chapters in background one by one"""
    if session_id not in manga_sessions:
        return
    
    session_data = manga_sessions[session_id]
    
    for chapter in session_data['chapters']:
        if chapter['status'] == 'pending' and 'raw_images' in chapter:
            process_chapter_background(session_id, chapter['id'], config)


@app.route("/reader/<session_id>/<int:chapter_id>")
def reader(session_id, chapter_id):
    """Manga reader page"""
    if session_id not in manga_sessions:
        return redirect("/")
    
    session_data = manga_sessions[session_id]
    chapters = session_data['chapters']
    
    # Find current chapter
    current_chapter = None
    for ch in chapters:
        if ch['id'] == chapter_id:
            current_chapter = ch
            break
    
    if not current_chapter:
        return redirect("/")
    
    # Create clean chapters list for JSON (without raw_images and numpy arrays)
    chapters_for_json = []
    for ch in chapters:
        chapters_for_json.append({
            'id': ch['id'],
            'name': ch['name'],
            'status': ch['status'],
            'progress': ch.get('progress', 0),
            'image_count': len(ch.get('images', []))
        })
    
    # Find prev/next chapters (use clean versions from chapters_for_json)
    current_idx = next((i for i, ch in enumerate(chapters_for_json) if ch['id'] == chapter_id), 0)
    prev_chapter = chapters_for_json[current_idx - 1] if current_idx > 0 else None
    next_chapter = chapters_for_json[current_idx + 1] if current_idx < len(chapters_for_json) - 1 else None
    
    # Create clean current_chapter (only include necessary data)
    current_chapter_clean = {
        'id': current_chapter['id'],
        'name': current_chapter['name'],
        'status': current_chapter['status'],
        'progress': current_chapter.get('progress', 0),
        'images': current_chapter.get('images', [])  # This is already base64 encoded
    }
    
    return render_template("reader.html",
        session_id=session_id,
        manga_title=session_data['title'],
        chapters=chapters_for_json,
        current_chapter=current_chapter_clean,
        prev_chapter=prev_chapter,
        next_chapter=next_chapter
    )


@app.route("/download_chapter_zip/<session_id>/<int:chapter_id>")
def download_chapter_zip(session_id, chapter_id):
    """Download a single chapter as ZIP"""
    if session_id not in manga_sessions:
        return redirect("/")
    
    session_data = manga_sessions[session_id]
    
    chapter = None
    for ch in session_data['chapters']:
        if ch['id'] == chapter_id:
            chapter = ch
            break
    
    if not chapter or chapter['status'] != 'completed':
        return redirect(f"/reader/{session_id}/{chapter_id}")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(chapter['images']):
            image_bytes = base64.b64decode(img['data'])
            filename = f"{img['name']}_translated.png"
            zip_file.writestr(filename, image_bytes)
    
    zip_buffer.seek(0)
    
    safe_name = re.sub(r'[^\w\-_\.]', '_', chapter['name'])
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{safe_name}_translated.zip"
    )


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Download all images as ZIP"""
    try:
        images_data = request.form.get("images_data", "[]")
        images = json.loads(images_data)
        
        if not images:
            return redirect("/")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(images):
                name = img.get('name', f'image_{i+1}')
                data = img.get('data', '')
                image_bytes = base64.b64decode(data)
                zip_file.writestr(f"{name}_translated.png", image_bytes)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_translated.zip'
        )
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return redirect("/")


if __name__ == "__main__":
    socketio.run(app, debug=True)
