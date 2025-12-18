"""
Helper functions for manga translation processing
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

from detect_bubbles import detect_bubbles
from process_bubble import process_bubble_auto
from add_text import add_text

MODEL_PATH = "model/model.pt"


def get_font_path(font_name: str) -> str:
    """Get the correct font file path based on font name."""
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


def process_images_with_batch(
    images_data,
    manga_translator,
    mocr,
    selected_font,
    translator_type,
    batch_size=10,
    use_context_memory=True,
    enable_black_bubble=True,
    socketio=None,
    chunk_size=16,
):
    """
    Process multiple images with multi-page batching.
    
    To tránh quá tải (upload hàng ngàn ảnh), hàm này CHIA THÀNH CÁC CHUNK NHỎ:
    - Mỗi chunk tối đa `chunk_size` ảnh (mặc định 16)
    - Với mỗi chunk: detect → OCR → dịch → render
    - Kết quả vẫn giữ nguyên THỨ TỰ ban đầu, không đảo lộn chapter/trang.
    """
    from translator.context_memory import ContextMemory

    def emit_progress(phase, current, total, message):
        if socketio:
            try:
                socketio.emit(
                    "progress",
                    {
                        "phase": phase,
                        "current": current,
                        "total": total,
                        "message": message,
                        "percent": int((current / max(total, 1)) * 100),
                    },
                )
            except:
                pass

    total_images = len(images_data)
    print(
        f"Processing {total_images} images (chunk_size={chunk_size})... "
        f"Context Memory: {'ON' if use_context_memory else 'OFF'}"
    )

    start_time = time.time()
    use_batch_ocr = hasattr(mocr, "process_batch")
    processed_results = []

    # Global counters for progress
    global_detect_idx = 0
    global_render_idx = 0

    # Process images in chunks to avoid huge prompts / payloads
    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_images = images_data[chunk_start:chunk_end]
        chunk_count = len(chunk_images)
        chunk_no = chunk_start // chunk_size + 1

        print(f"\n=== Chunk {chunk_no}: {chunk_count} images ({chunk_start+1}-{chunk_end}) ===")

        # Phase 1: Detect bubbles (per chunk)
        print("\n[Phase 1] Detecting bubbles...")
        emit_progress(
            "detection",
            global_detect_idx,
            total_images,
            f"Bắt đầu phát hiện speech bubbles (chunk {chunk_no})...",
        )

        all_pages_data = {}
        all_bubble_images = []
        bubble_mapping = []

        for idx, img_data in enumerate(chunk_images):
            image = img_data["image"]
            name = img_data["name"]

            global_detect_idx += 1
            emit_progress(
                "detection",
                global_detect_idx,
                total_images,
                f"Phát hiện bubbles: {name}",
            )
            print(f"  [{global_detect_idx}/{total_images}] {name}", end="", flush=True)

            results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
            if not results:
                all_pages_data[name] = {"image": image, "bubbles": [], "texts": []}
                print(" - 0 bubbles")
                continue

            print(f" - {len(results)} bubbles")

            bubble_data = []
            for bubble_idx, result in enumerate(results):
                if len(result) >= 7:
                    x1, y1, x2, y2, score, class_id, is_dark = result[:7]
                else:
                    x1, y1, x2, y2, score, class_id = result[:6]
                    is_dark = 0

                detected_image = image[int(y1) : int(y2), int(x1) : int(x2)]
                all_bubble_images.append(Image.fromarray(detected_image.copy()))
                bubble_mapping.append((name, bubble_idx))

                processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(
                    detected_image, force_dark=(is_dark == 1)
                )

                bubble_data.append(
                    {
                        "detected_image": processed_image,
                        "contour": cont,
                        "coords": (int(x1), int(y1), int(x2), int(y2)),
                        "is_dark": bubble_is_dark,
                        "fill_color": detected_color,
                    }
                )

            all_pages_data[name] = {
                "image": image,
                "bubbles": bubble_data,
                "texts": [],
            }

        # Phase 2: OCR (per chunk)
        if all_bubble_images:
            ocr_start = time.time()
            emit_progress(
                "ocr",
                0,
                1,
                f"Đang OCR {len(all_bubble_images)} bubbles (chunk {chunk_no})...",
            )
            print(
                f"\n[Phase 2] OCR processing {len(all_bubble_images)} bubbles...",
                end=" ",
                flush=True,
            )

            if use_batch_ocr:
                all_texts = mocr.process_batch(all_bubble_images)
            else:
                all_texts = [mocr(img) for img in all_bubble_images]

            for (page_name, bubble_idx), text in zip(bubble_mapping, all_texts):
                all_pages_data[page_name]["texts"].append(text)

            ocr_time = time.time() - ocr_start
            print(f"({ocr_time:.1f}s)")
            emit_progress(
                "ocr",
                1,
                1,
                f"OCR hoàn tất ({len(all_bubble_images)} bubbles, chunk {chunk_no})",
            )

        # Phase 3: Translation (per chunk)
        print("\n[Phase 3] Translation...")
        emit_progress("translation", 0, 1, "Đang dịch...")
        pages_texts = {
            name: data["texts"] for name, data in all_pages_data.items() if data["texts"]
        }
        all_translations = {}

        print(f"  Pages with text (chunk {chunk_no}): {len(pages_texts)}")
        print(f"  Translator type: {translator_type}")
        print(f"  Source: {manga_translator.source} -> Target: {manga_translator.target}")

        if pages_texts:
            translator = None

            if (
                translator_type == "copilot"
                and hasattr(manga_translator, "_local_llm_translator")
                and manga_translator._local_llm_translator
            ):
                translator = manga_translator._local_llm_translator
            elif (
                translator_type == "gemini"
                and hasattr(manga_translator, "_gemini_translator")
                and manga_translator._gemini_translator
            ):
                translator = manga_translator._gemini_translator
            elif (
                translator_type == "openrouter"
                and hasattr(manga_translator, "_openrouter_translator")
                and manga_translator._openrouter_translator
            ):
                translator = manga_translator._openrouter_translator

            if translator:
                context_memory = ContextMemory() if use_context_memory else None
                page_names = list(pages_texts.keys())

                for i in range(0, len(page_names), batch_size):
                    batch_names = page_names[i : i + batch_size]
                    batch_texts = {name: pages_texts[name] for name in batch_names}

                    try:
                        translated = translator.translate_pages_batch(
                            batch_texts,
                            source=manga_translator.source,
                            target=manga_translator.target,
                            context_memory=context_memory,
                        )
                        all_translations.update(translated)

                        if context_memory:
                            context_memory.update_from_batch(batch_texts, translated)
                    except Exception as e:
                        print(f"  ❌ Translation error: {e}")
                        for name, texts in batch_texts.items():
                            try:
                                all_translations[name] = translator.translate_batch(
                                    texts,
                                    manga_translator.source,
                                    manga_translator.target,
                                )
                            except Exception:
                                all_translations[name] = texts

        # Phase 4: Render text (per chunk)
        font_path = get_font_path(selected_font)
        print("\n[Phase 4] Rendering text...")

        for name, data in all_pages_data.items():
            global_render_idx += 1
            emit_progress(
                "rendering",
                global_render_idx,
                total_images,
                f"Render text: {name}",
            )

            image = data["image"]
            bubbles = data["bubbles"]
            original_texts = data["texts"]
            translated_texts = all_translations.get(name, original_texts)

            for bubble, text in zip(bubbles, translated_texts):
                x1, y1, x2, y2 = bubble["coords"]
                bubble_region = image[y1:y2, x1:x2]
                text_color = (255, 255, 255) if bubble.get("is_dark", False) else (0, 0, 0)

                add_text(bubble_region, text, font_path, bubble["contour"], text_color)

            processed_results.append({"image": image, "name": name})

    # Done all chunks
    total_time = time.time() - start_time
    avg_time = total_time / max(total_images, 1)

    print(f"{'='*50}")
    print(
        f"✓ TOTAL: {total_images} images processed in {total_time:.1f}s "
        f"({avg_time:.1f}s/image, chunk_size={chunk_size})"
    )
    print(f"{'='*50}\n")

    emit_progress(
        "done",
        total_images,
        total_images,
        f"Hoàn tất! {total_images} ảnh trong {total_time:.1f}s",
    )

    return processed_results
