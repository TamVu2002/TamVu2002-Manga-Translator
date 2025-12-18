"""
Gemini Translator with Batch Processing - OPTIMIZED
Uses Gemini 2.5 Flash-Lite for cost-effective translation
Better prompts for natural Vietnamese translation
With language validation for quality assurance
"""
import google.generativeai as genai
import json
import os
import time
from typing import List, Dict, Optional, TYPE_CHECKING

from .base import BaseTranslator
from .lang_validator import get_validator

if TYPE_CHECKING:
    from .context_memory import ContextMemory

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0  # 1s → 2s → 4s
BATCH_SIZE_LIMIT = 50   # Max texts per batch to avoid token limits


class GeminiTranslator(BaseTranslator):
    """
    Translator using Google Gemini 2.5 Flash-Lite.
    Optimized prompts for natural Vietnamese manga translation.
    """
    
    def __init__(self, api_key: str = None, custom_prompt: str = None, style: str = "default"):
        super().__init__(custom_prompt=custom_prompt, style=style)
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
    def _build_translation_prompt(self, source: str, target: str, style_override: str = None):
        """Build the core translation prompt based on target language."""
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style = style_override or self.custom_prompt
        style_text = f"\n\n[STYLE]: {style}" if style else ""
        
        if target == "vi":
            return f"""Bạn là chuyên gia dịch manga từ {source_name} sang Tiếng Việt.

🎯 NGUYÊN TẮC VÀNG:
1. Dịch như NGƯỜI VIỆT NÓI CHUYỆN - không phải robot dịch
2. Đọc to lên phải nghe tự nhiên, trôi chảy
3. Giữ cảm xúc và tính cách nhân vật

📝 HƯỚNG DẪN CHI TIẾT:

[TÊN RIÊNG]
- GIỮ NGUYÊN tên gốc: Tanaka, Sakura, Kim, Park, Lý, Trương...
- Kính ngữ Việt hóa tự nhiên:
  + -san/-kun → anh/chị/em hoặc bỏ qua nếu không cần
  + senpai → tiền bối, sensei → thầy/cô
  + oppa → anh, sunbae → tiền bối

[ĐẠI TỪ NHÂN XƯNG]
- Bạn bè thân thiết: tao/mày, tớ/cậu, mình/bạn
- Quan hệ bình thường: tôi/anh/chị/em/cậu
- Trang trọng/xa lạ: tôi/ngài/quý vị
- Gia đình: con/bố/mẹ/ông/bà/anh/chị/em
- Yêu đương: anh/em, mình/bạn

[THÁN TỪ & BIỂU CẢM]
- くそ/チクショウ/씨발 → Đ*t/Chết tiệt/Khốn kiếp
- やばい/대박 → Toang rồi/Xong đời/Chết mẹ
- すごい/대단해 → Đỉnh thật/Bá đạo/Điên thật
- なに/뭐 → Hả?/Cái gì?/Gì vậy?
- 大丈夫/괜찮아 → Ổn mà/Không sao/Được rồi
- え/어 → Ơ/Ủa/Hả
- ああ/아 → À/Ờ/Ừ

[KHẨU NGỮ TỰ NHIÊN]
- Dùng: oke, ngon, tởm, đỉnh, chill, toang, vãi, bá đạo, điên
- Rút gọn: ko (không), đc (được), j (gì), nc (nói chuyện)
- Từ đệm: à, ơi, nhỉ, nhé, đấy, đó, thôi, nha

[TRÁNH]
❌ Dịch từng chữ theo cấu trúc câu gốc
❌ Dùng quá nhiều từ Hán Việt học thuật
❌ Câu dài lê thê, thêm thắt không cần
❌ Giọng điệu robot, sách giáo khoa
❌ Dịch tên riêng ra nghĩa{style_text}

⚠️ CHỈ TRẢ VỀ BẢN DỊCH, KHÔNG GIẢI THÍCH."""
        else:
            return f"""You are an expert manga/comic translator from {source_name} to {target_name}.

RULES:
1. Translate for SPOKEN dialogue - natural when read aloud
2. Preserve character tone, emotion, and personality
3. Keep names in original form
4. Use natural sentence structures in {target_name}
5. Maintain impact of short/punchy lines{style_text}

Return ONLY the translation, no explanations."""
        
    def translate_single(
        self, 
        text: str, 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> str:
        """Translate a single text string."""
        if not text or not text.strip():
            return text
            
        base_prompt = self._build_translation_prompt(source, target, custom_prompt)
        prompt = f"{base_prompt}\n\nOriginal: {text}\n\nTranslation:"
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini translation error: {e}")
            return text
    
    def translate_batch(
        self, 
        texts: List[str], 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None
    ) -> List[str]:
        """Translate multiple texts in a single API call."""
        if not texts:
            return []
            
        # Filter empty texts
        indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed_texts:
            return texts
        
        texts_to_translate = [t for _, t in indexed_texts]
        
        # Split into smaller batches if needed
        if len(texts_to_translate) > BATCH_SIZE_LIMIT:
            all_translations = []
            for i in range(0, len(texts_to_translate), BATCH_SIZE_LIMIT):
                batch = texts_to_translate[i:i + BATCH_SIZE_LIMIT]
                batch_trans = self._translate_batch_internal(batch, source, target, custom_prompt)
                all_translations.extend(batch_trans)
            translations = all_translations
        else:
            translations = self._translate_batch_internal(texts_to_translate, source, target, custom_prompt)
        
        # Rebuild full list
        result = list(texts)
        for (orig_idx, _), trans in zip(indexed_texts, translations):
            result[orig_idx] = trans
            
        return result
    
    def _translate_batch_internal(
        self,
        texts_to_translate: List[str],
        source: str,
        target: str,
        custom_prompt: str = None
    ) -> List[str]:
        """Internal batch translation with retry logic."""
        base_prompt = self._build_translation_prompt(source, target, custom_prompt)
        
        prompt = f"""{base_prompt}

INPUT (JSON array - mỗi item là 1 bubble riêng biệt):
{json.dumps(texts_to_translate, ensure_ascii=False)}

OUTPUT: JSON array với bản dịch ĐÚNG THỨ TỰ. Ví dụ: ["dịch 1", "dịch 2", ...]
Chỉ trả về JSON array, không markdown, không giải thích."""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean up response
                result_text = self._clean_json_response(result_text)
                translations = json.loads(result_text)
                
                # Validate
                if len(translations) != len(texts_to_translate):
                    raise ValueError(f"Expected {len(texts_to_translate)}, got {len(translations)}")
                
                return translations
                
            except Exception as e:
                error_str = str(e)
                print(f"Gemini batch attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                
                # Quota exceeded - return originals immediately
                if "429" in error_str or "quota" in error_str.lower():
                    print("⚠️ Quota exceeded! Returning originals. Wait 1 min or upgrade plan.")
                    return texts_to_translate
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print("Falling back to single translations...")
                    return [self.translate_single(t, source, target) for t in texts_to_translate]
        
        return texts_to_translate
    
    def _clean_json_response(self, text: str) -> str:
        """Clean up JSON response from potential formatting issues."""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Try to find JSON array in response
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        
        return text
    
    def translate_pages_batch(
        self, 
        pages_texts: Dict[str, List[str]], 
        source: str = "ja", 
        target: str = "en",
        custom_prompt: str = None,
        context_memory: 'ContextMemory' = None
    ) -> Dict[str, List[str]]:
        """Translate texts from multiple pages in a single API call."""
        if not pages_texts:
            return {}
        
        base_prompt = self._build_translation_prompt(source, target, custom_prompt)
        
        # Build context section with consistency rules
        context_section = ""
        consistency_rules = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()
            consistency_rules = context_memory.generate_consistency_rules()
        
        prompt = f"""{base_prompt}
        
{context_section}
{consistency_rules}

📖 CÁC TRANG LIÊN TIẾP trong cùng 1 chapter.

⚠️ QUAN TRỌNG - TÍNH NHẤT QUÁN:
1. Một nhân vật phải LUÔN dùng cùng cách xưng hô với người khác
2. Giọng điệu (thô lỗ/lịch sự/thân mật) phải GIỮ NGUYÊN xuyên suốt
3. Tên riêng, biệt danh, kính ngữ phải THỐNG NHẤT
4. Nếu trang trước dùng "tao/mày", trang sau KHÔNG đổi sang "tôi/anh"
5. Context từ các trang trước quan trọng - tham khảo để giữ mạch truyện

INPUT (JSON - các trang liên tiếp):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

OUTPUT: JSON object với CẤU TRÚC GIỐNG HỆT, đã dịch.
Giữ nguyên tên page và thứ tự. Không giải thích, không markdown."""

        max_retries = 2
        validator = get_validator()
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                result_text = self._clean_json_response(result_text)
                
                # Try to find JSON object
                if '{' in result_text:
                    start = result_text.find('{')
                    end = result_text.rfind('}')
                    if start != -1 and end != -1:
                        result_text = result_text[start:end + 1]
                
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
                        print(f"   📊 Gemini validation: {len(validations) - failed_count}/{len(validations)} passed")
                        
                        if fail_rate > 40 and attempt < max_retries - 1:
                            print(f"   ⚠️ High failure rate ({fail_rate:.0f}%), retrying...")
                            time.sleep(1)
                            continue
                
                return translated
            
            except Exception as e:
                error_str = str(e).lower()
                print(f"Gemini pages batch error: {e}")
                
                if "429" in str(e) or "quota" in error_str:
                    print("⚠️ Quota exceeded!")
                    break
                
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        # Fallback: translate each page
        result = {}
        for page_name, texts in pages_texts.items():
            result[page_name] = self.translate_batch(texts, source, target)
        return result
