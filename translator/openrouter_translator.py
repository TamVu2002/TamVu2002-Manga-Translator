"""
OpenRouter Translator with Smart Fallback
Automatically switches models within provider when one fails
Includes language validation to ensure correct translation
"""
import requests
import json
import time
import re
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from .base import BaseTranslator

if TYPE_CHECKING:
    from .context_memory import ContextMemory

# Constants
MAX_RETRIES = 2
RETRY_DELAY = 1.0
BATCH_SIZE_LIMIT = 50
REQUEST_TIMEOUT_SINGLE = 60
REQUEST_TIMEOUT_BATCH = 120
REQUEST_TIMEOUT_PAGES = 240

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Language detection patterns
LANGUAGE_PATTERNS = {
    "ja": {
        "name": "Japanese",
        "patterns": [
            r'[\u3040-\u309F]',  # Hiragana
            r'[\u30A0-\u30FF]',  # Katakana
            r'[\u4E00-\u9FAF]',  # Kanji (shared with Chinese)
        ],
        "unique": r'[\u3040-\u309F\u30A0-\u30FF]',  # Hiragana + Katakana (unique to Japanese)
    },
    "ko": {
        "name": "Korean",
        "patterns": [r'[\uAC00-\uD7AF\u1100-\u11FF]'],  # Hangul
        "unique": r'[\uAC00-\uD7AF]',
    },
    "zh": {
        "name": "Chinese",
        "patterns": [r'[\u4E00-\u9FAF]'],  # CJK
        "unique": r'[\u4E00-\u9FAF]',  # Need context to distinguish from Japanese
    },
    "vi": {
        "name": "Vietnamese",
        "patterns": [
            r'[àáảãạăằắẳẵặâầấẩẫậ]',
            r'[èéẻẽẹêềếểễệ]',
            r'[ìíỉĩị]',
            r'[òóỏõọôồốổỗộơờớởỡợ]',
            r'[ùúủũụưừứửữự]',
            r'[ỳýỷỹỵđ]',
        ],
        "unique": r'[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]',
    },
    "en": {
        "name": "English",
        "patterns": [r'[a-zA-Z]'],
        "unique": r'\b(the|is|are|was|were|have|has|had|will|would|could|should|this|that|these|those|with|from|about|into|over|after|before|between|under|again|further|then|once)\b',
    },
}

# Minimum ratio of target language characters for validation
MIN_TARGET_LANG_RATIO = 0.15  # At least 15% of characters should be target language
MAX_SOURCE_LANG_RATIO = 0.30  # No more than 30% source language remaining


class OpenRouterTranslator(BaseTranslator):
    """
    Translator using OpenRouter API with smart model fallback.
    Automatically tries other models from same provider when one fails.
    """
    
    # Models organized by provider (priority order within each provider)
    # Updated Dec 2025 from OpenRouter API
    MODEL_PROVIDERS = {
        "claude": {
            "name": "Claude (Anthropic) 💰",
            "icon": "🟣",
            "requires_credits": True,
            "models": [
                ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5"),
                ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
                ("anthropic/claude-opus-4", "Claude Opus 4"),
            ]
        },
        "gpt": {
            "name": "GPT (OpenAI) 💰",
            "icon": "🟢",
            "requires_credits": True,
            "models": [
                ("openai/gpt-5.2-chat", "GPT-5.2 Chat"),
                ("openai/gpt-5.1-chat", "GPT-5.1 Chat"),
                ("openai/gpt-5.1", "GPT-5.1"),
            ]
        },
        "gemini": {
            "name": "Gemini (Google) 💰",
            "icon": "🔵",
            "requires_credits": True,
            "models": [
                ("google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
                ("google/gemini-2.5-flash", "Gemini 2.5 Flash"),
                ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
            ]
        },
        "gemma": {
            "name": "Gemma (Google) 💰",
            "icon": "💎",
            "requires_credits": True,
            "models": [
                ("google/gemma-2-27b-it", "Gemma 2 27B"),
                ("google/gemma-2-9b-it", "Gemma 2 9B"),
                ("google/gemma-7b-it", "Gemma 7B"),
                ("google/gemma-2-2b-it", "Gemma 2 2B"),
            ]
        },
        "deepseek": {
            "name": "DeepSeek 💰",
            "icon": "🔮",
            "requires_credits": True,
            "models": [
                ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
                ("deepseek/deepseek-r1", "DeepSeek R1"),
            ]
        },
        "free": {
            "name": "🆓 Free Models (Miễn phí!)",
            "icon": "🆓",
            "requires_credits": False,
            "models": [
                # Qwen models - TỐT NHẤT cho tiếng Việt (train trên multilingual data)
                ("qwen/qwen-2.5-72b-instruct:free", "Qwen 2.5 72B ⭐"),
                ("qwen/qwen-2.5-32b-instruct:free", "Qwen 2.5 32B ⭐"),
                ("qwen/qwen-2.5-14b-instruct:free", "Qwen 2.5 14B ⭐"),
                ("qwen/qwen-2.5-7b-instruct:free", "Qwen 2.5 7B ⭐"),
                # DeepSeek - tốt cho tiếng Á
                ("nex-agi/deepseek-v3.1-nex-n1:free", "DeepSeek V3.1 Nex"),
                # Nemotron - ổn định
                ("nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron Nano 30B"),
                # Fallback models
                ("amazon/nova-2-lite-v1:free", "Amazon Nova 2 Lite"),
                ("allenai/olmo-3-32b-think:free", "OLMo 3 32B"),
                ("mistralai/devstral-2512:free", "Devstral"),
                ("arcee-ai/trinity-mini:free", "Trinity Mini"),
            ]
        },
    }
    
    # Default provider (free because most users don't have credits)
    DEFAULT_PROVIDER = "free"
    
    def __init__(self, api_key: str, provider: str = None, custom_prompt: str = None, style: str = "default"):
        """
        Initialize OpenRouter translator with provider-based fallback.
        
        Args:
            api_key: OpenRouter API key
            provider: Provider name (claude, gpt, gemini, llama, mistral, deepseek, qwen, free)
            custom_prompt: Custom translation instructions
            style: Preset style
        """
        super().__init__(custom_prompt=custom_prompt, style=style)
        
        self.api_key = api_key
        self.provider = provider or self.DEFAULT_PROVIDER
        self.endpoint = f"{OPENROUTER_BASE_URL}/chat/completions"
        
        # Get models for this provider
        provider_data = self.MODEL_PROVIDERS.get(self.provider, self.MODEL_PROVIDERS[self.DEFAULT_PROVIDER])
        self.models = [m[0] for m in provider_data["models"]]
        self.provider_name = provider_data["name"]
        
        # Current model index (for fallback)
        self.current_model_idx = 0
        self.current_model = self.models[0] if self.models else "anthropic/claude-3.5-sonnet"
        
        # Track failed models
        self.failed_models = set()
        
        # Callback for UI notifications
        self.on_provider_exhausted = None
        
        # Headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://manga-translator.local",
            "X-Title": "Manga Translator"
        }
        
        # Validate API key format
        if not api_key or len(api_key) < 10 or not api_key.startswith("sk-or-"):
            print(f"⚠️ OpenRouter: Invalid API key")
            self.api_key = None
        else:
            print(f"✓ OpenRouter: {self.provider_name}")
        
        # Language validation stats
        self.validation_stats = {
            "total": 0,
            "passed": 0,
            "failed_source_detected": 0,
            "failed_target_missing": 0,
            "retried": 0,
        }
    
    # ============ LANGUAGE VALIDATION ============
    
    def _detect_language_ratio(self, text: str, lang_code: str) -> float:
        """
        Calculate ratio of characters matching a language pattern.
        Returns 0.0 to 1.0
        """
        if not text or lang_code not in LANGUAGE_PATTERNS:
            return 0.0
        
        lang_data = LANGUAGE_PATTERNS[lang_code]
        unique_pattern = lang_data.get("unique", lang_data["patterns"][0])
        
        # Count matching characters
        matches = re.findall(unique_pattern, text, re.IGNORECASE)
        
        # For word-based patterns (like English), count differently
        if lang_code == "en":
            total_words = len(text.split())
            return len(matches) / max(total_words, 1)
        
        # For character-based languages
        # Only count non-space, non-punctuation chars
        clean_text = re.sub(r'[\s\d\.,!?@#$%^&*()_+\-=\[\]{};\':"\\|<>\/~`]', '', text)
        total_chars = len(clean_text)
        
        return len(matches) / max(total_chars, 1)
    
    def _validate_translation(self, original: str, translated: str, source: str, target: str) -> Tuple[bool, str]:
        """
        Validate that translation is correct:
        1. Target language should be present
        2. Source language should be mostly gone (except names/terms)
        
        Returns: (is_valid, reason)
        """
        if not translated or not translated.strip():
            return False, "empty_response"
        
        # If original and translated are identical, likely not translated
        if original.strip() == translated.strip():
            return False, "unchanged"
        
        self.validation_stats["total"] += 1
        
        # Check source language ratio (should be low)
        source_ratio = self._detect_language_ratio(translated, source)
        
        # Check target language ratio (should be high)
        target_ratio = self._detect_language_ratio(translated, target)
        
        # Special case: Japanese to Vietnamese
        # Some kanji might remain as proper nouns - be lenient
        if source == "ja" and target == "vi":
            # Check specifically for hiragana/katakana (these should NOT remain)
            kana_ratio = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', translated)) / max(len(translated), 1)
            if kana_ratio > 0.15:
                self.validation_stats["failed_source_detected"] += 1
                return False, f"too_much_kana ({kana_ratio:.0%})"
        
        # General validation
        if source in ["ja", "ko", "zh"]:
            # For CJK sources, check if too much source remains
            if source_ratio > MAX_SOURCE_LANG_RATIO:
                self.validation_stats["failed_source_detected"] += 1
                return False, f"source_lang_high ({source_ratio:.0%})"
        
        # Special check: English to Vietnamese
        if source == "en" and target == "vi":
            # 1. Check for uppercase English sequences (I WILL ALWAYS NAKED)
            uppercase_english = re.findall(r'\b[A-Z]{2,}\b', translated)
            if len(uppercase_english) >= 2:  # 2+ uppercase English words = not translated
                self.validation_stats["failed_source_detected"] += 1
                return False, f"uppercase_english ({' '.join(uppercase_english[:3])})"
            
            # 2. Check for common English content words (not just function words)
            content_words = re.findall(r'\b(always|never|naked|wizard|book|found|mysterious|happened|cares|better|look|information|first|about|know|think|want|need|help|make|take|come|said|told|asked|answer|question|problem|really|actually|maybe|perhaps|something|nothing|everything|someone|anyone|everyone|somewhere|anywhere|everywhere)\b', translated.lower())
            if len(content_words) >= 2:  # 2+ content words in English = not translated
                self.validation_stats["failed_source_detected"] += 1
                return False, f"english_content_words ({len(content_words)} words)"
            
            # 3. Count common English function words
            function_words = re.findall(r'\b(the|is|are|was|were|have|has|had|will|would|could|should|this|that|what|where|when|why|how|but|and|or|not|you|your|my|me|he|she|it|we|they|for|with|from|about|after|before|into|through|during|because|while|although|however|therefore|then|now|just|only|also|very|too|more|most|some|any|all|each|every|both|few|many|much|other|another|such|no|yes|here|there|who|which)\b', translated.lower())
            english_word_count = len(function_words)
            word_count = len(translated.split())
            
            # If more than 30% are English function words, reject (reduced from 40%)
            if word_count > 3 and english_word_count / word_count > 0.3:
                self.validation_stats["failed_source_detected"] += 1
                return False, f"too_much_english ({english_word_count}/{word_count} words)"
        
        # Check target language presence
        if target == "vi":
            # Vietnamese should have diacritics
            if target_ratio < MIN_TARGET_LANG_RATIO:
                # Exception: very short text might not have diacritics
                if len(translated) > 20:
                    self.validation_stats["failed_target_missing"] += 1
                    return False, f"target_lang_low ({target_ratio:.0%})"
            
            # Quality checks for Vietnamese
            # 1. Check for overly formal/robotic patterns
            robotic_patterns = [
                r'\btôi\s+(?:đã\s+)?tìm\s+thấy\b',  # "tôi tìm thấy" (too formal)
                r'\btôi\s+(?:đã\s+)?nhận\s+thấy\b',  # "tôi nhận thấy"
                r'\bđiều\s+này\s+có\s+thể\b',  # "điều này có thể" (too formal)
            ]
            for pattern in robotic_patterns:
                if re.search(pattern, translated, re.IGNORECASE):
                    self.validation_stats["failed_target_missing"] += 1
                    return False, "too_formal_robotic"
            
            # 2. Check for overly long sentences (sign of word-by-word translation)
            sentences = re.split(r'[.!?]+', translated)
            for sent in sentences:
                words = sent.split()
                if len(words) > 25:  # Very long sentence = likely machine translation
                    self.validation_stats["failed_target_missing"] += 1
                    return False, "sentence_too_long"
            
            # 3. Check for too many Hán-Việt words (unnatural)
            han_viet_patterns = [
                r'\b(?:thực\s+hiện|tiến\s+hành|thực\s+tế|hiện\s+tại|quan\s+trọng|chính\s+xác)\b'
            ]
            han_viet_count = sum(len(re.findall(pattern, translated, re.IGNORECASE)) for pattern in han_viet_patterns)
            word_count = len(translated.split())
            if word_count > 5 and han_viet_count / word_count > 0.3:
                self.validation_stats["failed_target_missing"] += 1
                return False, "too_many_han_viet"
        
        self.validation_stats["passed"] += 1
        return True, "ok"
    
    def _validate_batch(self, originals: List[str], translations: List[str], source: str, target: str) -> Tuple[List[bool], List[str]]:
        """Validate a batch of translations."""
        results = []
        reasons = []
        
        for orig, trans in zip(originals, translations):
            valid, reason = self._validate_translation(orig, trans, source, target)
            results.append(valid)
            reasons.append(reason)
        
        return results, reasons
    
    def get_validation_stats(self) -> Dict:
        """Get validation statistics."""
        total = self.validation_stats["total"]
        if total == 0:
            return {"pass_rate": "N/A", **self.validation_stats}
        
        pass_rate = self.validation_stats["passed"] / total * 100
        return {
            "pass_rate": f"{pass_rate:.1f}%",
            **self.validation_stats
        }
    
    def _get_next_model(self) -> Optional[str]:
        """Get next available model in the provider, skipping failed ones."""
        for i, model in enumerate(self.models):
            if model not in self.failed_models:
                self.current_model_idx = i
                self.current_model = model
                return model
        return None
    
    def _mark_model_failed(self, model: str):
        """Mark a model as failed."""
        self.failed_models.add(model)
        model_name = self._get_model_name(model)
        print(f"   ❌ Model failed: {model_name}")
    
    def _get_model_name(self, model_id: str) -> str:
        """Get human-readable model name."""
        for provider_data in self.MODEL_PROVIDERS.values():
            for mid, name in provider_data["models"]:
                if mid == model_id:
                    return name
        return model_id
    
    def _check_all_models_failed(self) -> bool:
        """Check if all models in provider have failed."""
        return len(self.failed_models) >= len(self.models)
    
    def get_fallback_suggestions(self) -> List[str]:
        """Get suggested alternative providers."""
        suggestions = []
        for key, data in self.MODEL_PROVIDERS.items():
            if key != self.provider:
                suggestions.append(f"{data['icon']} {data['name']}")
        return suggestions[:3]  # Return top 3 suggestions
    
    def reset_failed_models(self):
        """Reset failed models (useful when switching providers)."""
        self.failed_models.clear()
        self.current_model_idx = 0
        self.current_model = self.models[0] if self.models else None
    
    def switch_provider(self, new_provider: str):
        """Switch to a different provider."""
        if new_provider in self.MODEL_PROVIDERS:
            self.provider = new_provider
            provider_data = self.MODEL_PROVIDERS[new_provider]
            self.models = [m[0] for m in provider_data["models"]]
            self.provider_name = provider_data["name"]
            self.reset_failed_models()
            print(f"🔄 Switched to {self.provider_name}")
    
    def _build_translation_prompt(self, source: str, target: str, style_override: str = None):
        """Build translation prompt."""
        source_name = self.LANG_NAMES.get(source, "Japanese")
        target_name = self.LANG_NAMES.get(target, "English")
        style = style_override or self.custom_prompt
        style_text = f"\n\n[STYLE]: {style}" if style else ""
        
        if target == "vi":
            return f"""Bạn là NGƯỜI DỊCH TRUYỆN TRANH CHUYÊN NGHIỆP, không phải AI máy móc.

🎭 MỤC TIÊU: Dịch như một người Việt đọc truyện tranh thật sự - TỰ NHIÊN, CÓ CẢM XÚC, DỄ HIỂU.

⚠️ BẮT BUỘC: 
- Output 100% TIẾNG VIỆT - KHÔNG giữ nguyên tiếng Anh
- Dịch theo NGỮ CẢNH của câu chuyện, không dịch word-by-word
- Giữ LIÊN KẾT giữa các câu - đọc phải MẠCH LẠC
- Hiểu TÍNH CÁCH nhân vật để dịch đúng giọng điệu

📖 CÁCH DỊCH TỰ NHIÊN:

VÍ DỤ 1 - Câu đơn:
❌ Máy móc: "I found a mysterious book" → "Tôi tìm thấy một cuốn sách bí ẩn"
✅ Tự nhiên: "I found a mysterious book" → "Tao tìm được một cuốn sách bí ẩn"

VÍ DỤ 2 - Câu cảm thán:
❌ Máy móc: "What exactly happened?" → "Điều gì đã xảy ra chính xác?"
✅ Tự nhiên: "What exactly happened?" → "Chuyện gì xảy ra thế?"

VÍ DỤ 3 - Câu ngắn, cảm xúc:
❌ Máy móc: "But who cares!!" → "Nhưng ai quan tâm!!"
✅ Tự nhiên: "But who cares!!" → "Nhưng kệ đi!!"

VÍ DỤ 4 - Câu dài, có logic:
❌ Máy móc: "Better I look for information first" → "Tốt hơn tôi tìm thông tin trước"
✅ Tự nhiên: "Better I look for information first" → "Tốt hơn hết là tao tìm thông tin trước đã"

🎯 QUY TẮC DỊCH NHƯ NGƯỜI:

1. ĐỌC HIỂU TRƯỚC KHI DỊCH:
   - Hiểu ngữ cảnh: nhân vật đang làm gì? Cảm xúc ra sao?
   - Hiểu mối quan hệ: họ thân thiết hay xa cách?
   - Hiểu tình huống: đang vui, buồn, giận, sợ?

2. DỊCH THEO TÍNH CÁCH:
   - Nhân vật thân thiết → dùng "tao/mày", "tớ/cậu"
   - Nhân vật lịch sự → dùng "tôi/anh/chị"
   - Nhân vật trẻ con → dùng "con/bố/mẹ"
   - Nhân vật yêu nhau → dùng "anh/em"

3. GIỮ LIÊN KẾT:
   - Câu trước nói về A, câu sau nói về A → dùng đại từ "nó", "cái đó"
   - Câu trước hỏi, câu sau trả lời → phải khớp logic
   - Các câu trong cùng trang → phải có mạch truyện

4. TỰ NHIÊN NHƯ NÓI:
   - Người Việt nói ngắn gọn → dịch ngắn gọn
   - Người Việt dùng từ đời thường → dùng từ đời thường
   - Tránh câu dài lê thê, từ Hán Việt khó hiểu

[ĐẠI TỪ theo mối quan hệ]
- Bạn thân: tao/mày, tớ/cậu
- Người lạ: tôi/anh/chị/em
- Gia đình: con/bố/mẹ/anh/chị
- Yêu nhau: anh/em
- Cấp trên: thầy/cô, tiền bối

[THÁN TỪ tự nhiên]
- Oh/Wow → Ồ/Chà/Ơ
- What → Cái gì/Hả/Gì đây
- Damn → Chết tiệt/Đồ quỷ/Đ*t
- Amazing → Tuyệt vời/Đỉnh/Bá đạo
- Really? → Thật à?/Thật không?

[TÊN RIÊNG] Giữ nguyên: Tanaka, Sakura, Kim, Park...

❌ TRÁNH:
- Dịch word-by-word (từng từ một)
- Câu dài lê thê, khó đọc
- Từ Hán Việt nhiều, không tự nhiên
- Giữ nguyên tiếng Anh
- Mất liên kết giữa các câu

✅ LÀM:
- Dịch theo ngữ cảnh
- Câu ngắn gọn, dễ hiểu
- Tự nhiên như người Việt nói
- Giữ mạch truyện liên tục{style_text}"""
        else:
            return f"""Expert manga translator from {source_name} to {target_name}.

RULES:
1. Natural spoken dialogue
2. Keep character personality/emotion  
3. Keep original names
4. Short lines stay impactful{style_text}"""

    def _make_request_with_fallback(self, prompt: str, timeout: int = REQUEST_TIMEOUT_SINGLE) -> str:
        """Make API request with automatic model fallback."""
        last_error = None
        
        if not self.api_key or not self.api_key.startswith("sk-or-"):
            raise Exception("Invalid OpenRouter API key")
        
        while True:
            model = self._get_next_model()
            
            if model is None:
                suggestions = self.get_fallback_suggestions()
                if self.on_provider_exhausted:
                    self.on_provider_exhausted(self.provider, suggestions)
                raise Exception(f"All models in {self.provider_name} failed")
            
            model_name = self._get_model_name(model)
            
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(
                        self.endpoint,
                        headers=self.headers,
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You are a professional Vietnamese manga translator. Your translations must be:\n1. 100% in Vietnamese - never keep English text\n2. Natural and fluent - like a native Vietnamese reader\n3. Context-aware - understand the story flow and character relationships\n4. Emotionally accurate - preserve character personality and tone\n5. Coherent - maintain logical connections between sentences and pages"},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.7,  # Higher for more natural, creative translations
                            "max_tokens": 4096,
                        },
                        timeout=timeout
                    )
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        wait_time = 10 + (attempt * 10)
                        print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        self._mark_model_failed(self.current_model)
                        break  # Try next model
                    
                    if response.status_code >= 400:
                        error_data = response.json() if response.text else {}
                        error_msg = error_data.get("error", {}).get("message", response.text[:200])
                        raise Exception(f"HTTP {response.status_code}: {error_msg}")
                    
                    result = response.json()
                    if "error" in result:
                        raise Exception(result["error"].get("message", "Unknown error"))
                    
                    content = result["choices"][0]["message"]["content"].strip()
                    if not content:
                        raise Exception("Empty response")
                    
                    return content
                    
                except requests.exceptions.Timeout:
                    last_error = f"Timeout"
                    
                except Exception as e:
                    last_error = str(e)
                    # Model-specific errors: try next model
                    if any(x in last_error.lower() for x in ["model", "not found", "unavailable", "rate", "quota", "credit"]):
                        break
                    # Transient error: retry
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        break
            
            self._mark_model_failed(model)

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
        
        # Find JSON
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
        """Translate single text with fallback and validation."""
        if not text or not text.strip():
            return text
        
        base_prompt = self._build_translation_prompt(source, target)
        
        for attempt in range(3):  # Increased retries for quality
            try:
                if attempt == 0:
                    prompt = f"{base_prompt}\n\nOriginal: {text}\n\nChỉ trả về bản dịch:"
                else:
                    # Enhanced prompt on retry - emphasize natural translation
                    quality_reminder = """
⚠️ LƯU Ý QUAN TRỌNG:
- Dịch TỰ NHIÊN như người Việt nói, KHÔNG máy móc
- Dùng từ đời thường: tao/mày, tớ/cậu (không dùng "tôi tìm thấy")
- Câu ngắn gọn, dễ hiểu
- Giữ cảm xúc và tính cách nhân vật
"""
                    prompt = f"{base_prompt}{quality_reminder}\n\nOriginal: {text}\n\nChỉ trả về bản dịch TỰ NHIÊN:"
                
                result = self._make_request_with_fallback(prompt, REQUEST_TIMEOUT_SINGLE)
                is_valid, reason = self._validate_translation(text, result, source, target)
                
                if is_valid:
                    return result
                
                # If validation failed
                if attempt == 2:
                    # Last attempt - if still invalid, return original (better than wrong translation)
                    print(f"   ⚠️ All attempts failed validation ({reason}), keeping original text")
                    return text
                
                print(f"   ⚠️ Validation failed ({reason}), retrying with different model...")
                self.validation_stats["retried"] += 1
                self._mark_model_failed(self.current_model)
                    
            except Exception as e:
                if attempt == 2:
                    return text
        
        return text
    
    def translate_batch(self, texts: List[str], source: str = "ja", target: str = "en") -> List[str]:
        """Translate batch with fallback."""
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
        """Internal batch translation with fallback and validation."""
        base_prompt = self._build_translation_prompt(source, target)
        
        prompt = f"""{base_prompt}

INPUT (JSON array - cần dịch):
{json.dumps(texts_to_translate, ensure_ascii=False)}

⚠️ OUTPUT: JSON array TIẾNG VIỆT. Mỗi câu phải dịch HOÀN TOÀN sang tiếng Việt!
Ví dụ: ["Tao tìm được rồi!", "Chuyện gì thế?", "Đi thôi!"]
Chỉ JSON, không giải thích."""

        for attempt in range(3):  # Increased to 3 attempts
            try:
                result_text = self._make_request_with_fallback(prompt, REQUEST_TIMEOUT_BATCH)
                result_text = self._clean_json_response(result_text)
                translations = json.loads(result_text)
                
                # Fix length mismatch
                while len(translations) < len(texts_to_translate):
                    translations.append(texts_to_translate[len(translations)])
                translations = translations[:len(texts_to_translate)]
                
                # Validate translations
                validations, reasons = self._validate_batch(texts_to_translate, translations, source, target)
                failed_count = validations.count(False)
                fail_rate = failed_count / len(validations) if validations else 0
                
                if failed_count == 0:
                    return translations
                
                # Too many failures - retry (lowered threshold to 15%)
                if fail_rate > 0.15 and attempt < 2:
                    print(f"   ⚠️ Quality check: {failed_count}/{len(validations)} failed ({fail_rate:.0%}) - retrying...")
                    self._mark_model_failed(self.current_model)
                    continue
                
                # Re-translate ALL failed items individually for better quality
                if failed_count > 0:
                    print(f"   🔄 Re-translating {failed_count} failed items individually...")
                    for i, (valid, reason) in enumerate(zip(validations, reasons)):
                        if not valid:
                            print(f"      Item {i+1}: {reason}")
                            translations[i] = self.translate_single(texts_to_translate[i], source, target)
                
                return translations
                
            except json.JSONDecodeError:
                return [self.translate_single(t, source, target) for t in texts_to_translate]
            except Exception:
                return texts_to_translate
        
        return texts_to_translate
    
    def translate_pages_batch(
        self, 
        pages_texts: Dict[str, List[str]], 
        source: str = "ja", 
        target: str = "en",
        context_memory: 'ContextMemory' = None
    ) -> Dict[str, List[str]]:
        """Translate multiple pages with context consistency."""
        if not pages_texts:
            return {}
        
        print(f"   📖 Translating {len(pages_texts)} pages ({source} → {target})...")
        
        base_prompt = self._build_translation_prompt(source, target)
        
        # Add context if available
        context_section = ""
        consistency_rules = ""
        if context_memory:
            context_section = context_memory.generate_context_prompt()
            consistency_rules = context_memory.generate_consistency_rules()
        
        for attempt in range(3):  # Increased retries for quality
            # Enhanced prompt with story understanding
            if attempt == 0:
                story_context = f"""

{context_section}
{consistency_rules}

📖 ĐÂY LÀ CÁC TRANG LIÊN TIẾP trong cùng chapter manga.

🎭 QUAN TRỌNG - HIỂU CỐT TRUYỆN:
1. ĐỌC TẤT CẢ các trang TRƯỚC KHI dịch - hiểu cốt truyện đang diễn ra
2. Các trang LIÊN KẾT với nhau - nhân vật, sự kiện, cảm xúc phải MẠCH LẠC
3. Dịch như bạn ĐÃ ĐỌC HẾT chapter này rồi - không dịch từng trang riêng lẻ

⚠️ BẮT BUỘC:
1. Dịch TẤT CẢ sang TIẾNG VIỆT - KHÔNG để nguyên tiếng Anh
2. Nhất quán cách xưng hô (tao/mày, tớ/cậu...) XUYÊN SUỐT tất cả trang
3. Giữ cảm xúc, giọng điệu nhân vật - phải GIỐNG NHAU giữa các trang
4. Dịch TỰ NHIÊN - như người Việt đọc truyện, KHÔNG máy móc

📥 INPUT (cần dịch sang tiếng Việt):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

📤 OUTPUT: JSON object với TẤT CẢ text đã dịch sang TIẾNG VIỆT TỰ NHIÊN. Không giải thích.
Ví dụ format: {{"page1": ["Câu tiếng Việt 1", "Câu tiếng Việt 2"], ...}}"""
            else:
                # Retry with stronger emphasis on quality
                story_context = f"""

{context_section}
{consistency_rules}

📖 ĐÂY LÀ CÁC TRANG LIÊN TIẾP - DỊCH LẠI CHO TỰ NHIÊN HƠN!

🚨 LẦN TRƯỚC DỊCH CHƯA TỰ NHIÊN - CẦN DỊCH LẠI:
- ĐỌC HẾT tất cả trang để HIỂU CỐT TRUYỆN
- Dịch TỰ NHIÊN như người Việt nói (tao/mày, không phải "tôi tìm thấy")
- Giữ LIÊN KẾT giữa các trang - nhân vật, sự kiện phải MẠCH LẠC
- Câu ngắn gọn, dễ hiểu - TRÁNH câu dài lê thê

⚠️ BẮT BUỘC:
1. Dịch TẤT CẢ sang TIẾNG VIỆT - KHÔNG để nguyên tiếng Anh
2. Nhất quán cách xưng hô XUYÊN SUỐT tất cả trang
3. Dịch TỰ NHIÊN - như người Việt đọc truyện

📥 INPUT (cần dịch LẠI cho TỰ NHIÊN):
{json.dumps(pages_texts, ensure_ascii=False, indent=2)}

📤 OUTPUT: JSON object với TẤT CẢ text đã dịch sang TIẾNG VIỆT TỰ NHIÊN. Không giải thích."""

            prompt = f"""{base_prompt}{story_context}"""
            try:
                result_text = self._make_request_with_fallback(prompt, REQUEST_TIMEOUT_PAGES)
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
                    validations, reasons = self._validate_batch(all_originals, all_translations, source, target)
                    failed_count = validations.count(False)
                    fail_rate = failed_count / len(validations) if validations else 0
                    
                    # Stricter validation - retry if quality is poor
                    if fail_rate > 0.2 and attempt < 2:  # Lower threshold, more retries
                        print(f"   ⚠️ Quality check: {failed_count}/{len(validations)} failed ({fail_rate:.0%}) - retrying...")
                        self.validation_stats["retried"] += 1
                        if attempt == 0:
                            self._mark_model_failed(self.current_model)
                        continue
                    
                    # Re-translate failed items individually for better quality
                    if failed_count > 0 and failed_count < len(validations) / 2:
                        print(f"   🔄 Re-translating {failed_count} low-quality items...")
                        idx = 0
                        for page_name in pages_texts:
                            if page_name in translated:
                                for j in range(len(pages_texts[page_name])):
                                    if idx < len(validations) and not validations[idx]:
                                        translated[page_name][j] = self.translate_single(
                                            all_originals[idx], source, target
                                        )
                                    idx += 1
                
                print(f"   ✅ Translated {len(pages_texts)} pages")
                return translated
                
            except json.JSONDecodeError:
                # Fallback to batch translation
                result = {}
                for page_name, texts in pages_texts.items():
                    result[page_name] = self.translate_batch(texts, source, target)
                return result
            except Exception:
                if attempt == 0:
                    self._mark_model_failed(self.current_model)
                    continue
                return pages_texts
        
        return pages_texts
    
    def get_status(self) -> Dict:
        """Get current translator status."""
        return {
            "provider": self.provider,
            "provider_name": self.provider_name,
            "current_model": self._get_model_name(self.current_model),
            "failed_models": [self._get_model_name(m) for m in self.failed_models],
            "available_models": len(self.models) - len(self.failed_models),
            "total_models": len(self.models),
        }
    
    @classmethod
    def get_providers(cls) -> Dict:
        """Get all available providers for UI."""
        return {
            key: {
                "name": data["name"],
                "icon": data["icon"],
                "model_count": len(data["models"])
            }
            for key, data in cls.MODEL_PROVIDERS.items()
        }
