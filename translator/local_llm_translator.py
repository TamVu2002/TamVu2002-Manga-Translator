"""
Local LLM Translator - OPTIMIZED
Uses OpenAI-compatible API endpoints (Ollama, LM Studio, LocalAI, vLLM, Copilot-API, etc.)
Better prompts for natural Vietnamese translation
With language validation for quality assurance
"""
import requests
import json
import time
from typing import List, Dict, TYPE_CHECKING

from .base import BaseTranslator
from .lang_validator import get_validator

if TYPE_CHECKING:
    from .context_memory import ContextMemory

# Constants
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0
BATCH_SIZE_LIMIT = 40
REQUEST_TIMEOUT_SINGLE = 30
REQUEST_TIMEOUT_BATCH = 90
REQUEST_TIMEOUT_PAGES = 180


class LocalLLMTranslator(BaseTranslator):
    """
    Translator using OpenAI-compatible local LLM servers.
    Works with Ollama, LM Studio, LocalAI, vLLM, Copilot-API.
    """
    
    # Available models (from Copilot API)
    MODELS = [
        "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.1-codex",
        "gpt-4.1", "gpt-41-copilot",
        "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20",
        "gpt-4", "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "claude-sonnet-4.5", "claude-sonnet-4", "claude-opus-4.5", "claude-haiku-4.5",
        "gemini-3-pro-preview", "gemini-2.5-pro",
        "grok-code-fast-1",
    ]
    
    def __init__(self, server_url: str = "http://localhost:8080", model: str = "gpt-4o", 
                 custom_prompt: str = None, style: str = "default"):
        super().__init__(custom_prompt=custom_prompt, style=style)
        
        self.base_url = server_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/v1/chat/completions"
    
    def _build_translation_prompt(self, source: str, target: str, style_override: str = None):
        """Build the core translation prompt based on target language."""
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style = style_override or self.custom_prompt
        style_text = f"\n\n[STYLE]: {style}" if style else ""
        
        if target == "vi":
            return f"""Bạn là chuyên gia dịch manga từ {source_name} sang Tiếng Việt.

🎯 NGUYÊN TẮC:
1. Dịch như NGƯỜI VIỆT NÓI - tự nhiên, trôi chảy
2. Giữ cảm xúc và tính cách nhân vật
3. Đọc to phải nghe hay

📝 HƯỚNG DẪN:

[TÊN] Giữ nguyên: Tanaka, Sakura, Kim, Park, Lý...
- -san/-kun → anh/chị/em hoặc bỏ
- senpai/sunbae → tiền bối
- sensei → thầy/cô, oppa → anh

[ĐẠI TỪ]
- Thân: tao/mày, tớ/cậu
- Bình thường: tôi/anh/em
- Gia đình: con/bố/mẹ
- Yêu: anh/em

[THÁN TỪ]
- くそ → Đ*t/Chết tiệt
- やばい → Toang/Xong đời
- すごい → Đỉnh/Bá đạo
- なに → Hả/Cái gì

[DÙNG] oke, ngon, tởm, đỉnh, chill, toang, vãi

[TRÁNH] ❌ Dịch word-by-word ❌ Từ Hán Việt nhiều ❌ Câu dài lê thê ❌ Giọng robot{style_text}"""
        else:
            return f"""Expert manga translator from {source_name} to {target_name}.

RULES:
1. Natural spoken dialogue
2. Keep character personality/emotion
3. Keep original names
4. Short lines stay impactful{style_text}"""

    def _make_request(self, prompt: str, timeout: int = REQUEST_TIMEOUT_SINGLE) -> str:
        """Make API request with error handling."""
        response = requests.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response."""
        text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Find JSON array/object
        if text.startswith('[') or text.startswith('{'):
            return text
        
        start_arr = text.find('[')
        start_obj = text.find('{')
        
        if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
            end = text.rfind(']')
            if end != -1:
                return text[start_arr:end + 1]
        elif start_obj != -1:
            end = text.rfind('}')
            if end != -1:
                return text[start_obj:end + 1]
        
        return text

    def translate_single(self, text: str, source: str = "ja", target: str = "en") -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return text
        
        base_prompt = self._build_translation_prompt(source, target)
        prompt = f"{base_prompt}\n\nOriginal: {text}\n\nChỉ trả về bản dịch:"
        
        try:
            return self._make_request(prompt, REQUEST_TIMEOUT_SINGLE)
        except Exception as e:
            print(f"LLM translation error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source: str = "ja", target: str = "en") -> List[str]:
        """Translate multiple texts in a single API call."""
        if not texts:
            return []
        
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed_texts:
            return texts
        
        texts_to_translate = [t for _, t in indexed_texts]
        
        # Split large batches
        if len(texts_to_translate) > BATCH_SIZE_LIMIT:
            all_translations = []
            for i in range(0, len(texts_to_translate), BATCH_SIZE_LIMIT):
                batch = texts_to_translate[i:i + BATCH_SIZE_LIMIT]
                batch_trans = self._translate_batch_internal(batch, source, target)
                all_translations.extend(batch_trans)
            translations = all_translations
        else:
            translations = self._translate_batch_internal(texts_to_translate, source, target)
        
        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans
        
        return result
    
    def _translate_batch_internal(self, texts_to_translate: List[str], source: str, target: str) -> List[str]:
        """Internal batch translation with retry."""
        base_prompt = self._build_translation_prompt(source, target)
        
        prompt = f"""{base_prompt}

INPUT (JSON array):
{json.dumps(texts_to_translate, ensure_ascii=False)}

OUTPUT: JSON array bản dịch ĐÚNG THỨ TỰ. VD: ["dịch 1", "dịch 2"]
Chỉ JSON, không giải thích."""

        for attempt in range(MAX_RETRIES):
            try:
                result_text = self._make_request(prompt, REQUEST_TIMEOUT_BATCH)
                result_text = self._clean_json_response(result_text)
                translations = json.loads(result_text)
                
                # Validate and fix length
                if len(translations) != len(texts_to_translate):
                    print(f"Warning: Expected {len(texts_to_translate)}, got {len(translations)}")
                    while len(translations) < len(texts_to_translate):
                        translations.append(texts_to_translate[len(translations)])
                    translations = translations[:len(texts_to_translate)]
                
                return translations
                
            except Exception as e:
                print(f"LLM batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                else:
                    print("Falling back to single translations...")
                    return [self.translate_single(t, source, target) for t in texts_to_translate]
        
        return texts_to_translate
    
    def translate_pages_batch(
        self, 
        pages_texts: dict, 
        source: str = "ja", 
        target: str = "en",
        context_memory: 'ContextMemory' = None
    ) -> dict:
        """Translate texts from multiple pages in one call."""
        if not pages_texts:
            return {}
        
        base_prompt = self._build_translation_prompt(source, target)
        
        # Build context with consistency rules
        context_section = ""
        consistency_rules = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()
            consistency_rules = context_memory.generate_consistency_rules()
        
        prompt = f"""{base_prompt}

{context_section}
{consistency_rules}

📖 CÁC TRANG LIÊN TIẾP trong cùng chapter.

⚠️ QUAN TRỌNG - NHẤT QUÁN:
1. Mỗi nhân vật LUÔN dùng cùng cách xưng hô
2. Giọng điệu GIỮ NGUYÊN xuyên suốt (đã thô lỗ → vẫn thô lỗ)
3. Tên riêng, biệt danh THỐNG NHẤT
4. Trang trước dùng "tao/mày" → trang sau KHÔNG đổi "tôi/anh"

INPUT:
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

OUTPUT: JSON object CẤU TRÚC GIỐNG, đã dịch. Không giải thích."""

        validator = get_validator()
        
        for attempt in range(MAX_RETRIES):
            try:
                
                result_text = self._make_request(prompt, REQUEST_TIMEOUT_PAGES)
                result_text = self._clean_json_response(result_text)
                translated = json.loads(result_text)
                
                # Validate translations
                all_originals = []
                all_translations = []
                
                for page_name in pages_texts:
                    if page_name in translated:
                        for orig, trans in zip(pages_texts[page_name], translated[page_name]):
                            all_originals.append(orig)
                            all_translations.append(trans)
                
                if all_originals:
                    validations, reasons = validator.validate_batch(all_originals, all_translations, source, target)
                    fail_rate = validator.get_failure_rate(validations)
                    
                    if fail_rate > 0:
                        failed_count = validations.count(False)
                        print(f"   📊 LLM validation: {len(validations) - failed_count}/{len(validations)} passed")
                        
                        if fail_rate > 40 and attempt < MAX_RETRIES - 1:
                            print(f"   ⚠️ High failure rate ({fail_rate:.0f}%), retrying...")
                            time.sleep(RETRY_DELAY_BASE)
                            continue
                
                print(f"✓ Translated {len(pages_texts)} pages in batch")
                return translated
                
            except Exception as e:
                print(f"LLM pages batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                else:
                    print("Falling back to single-page batches...")
                    result = {}
                    for page_name, texts in pages_texts.items():
                        result[page_name] = self.translate_batch(texts, source, target)
                    return result
        
        return pages_texts
    
    def test_connection(self) -> bool:
        """Test if the server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from server."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
        except:
            pass
        return self.MODELS
