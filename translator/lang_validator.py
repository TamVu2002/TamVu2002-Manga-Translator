"""
Language Validation Module
Validates translations to ensure correct source-to-target conversion
"""
import re
from typing import Dict, List, Tuple

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
        "kana_only": r'[\u3040-\u309F\u30A0-\u30FF]',
    },
    "ko": {
        "name": "Korean",
        "patterns": [r'[\uAC00-\uD7AF\u1100-\u11FF]'],  # Hangul
        "unique": r'[\uAC00-\uD7AF]',
    },
    "zh": {
        "name": "Chinese",
        "patterns": [r'[\u4E00-\u9FAF]'],  # CJK
        "unique": r'[\u4E00-\u9FAF]',
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
        "unique": r'\b(the|is|are|was|were|have|has|had|will|would|could|should|this|that|with|from|about)\b',
    },
}

# Thresholds
MIN_TARGET_LANG_RATIO = 0.10  # At least 10% target language chars
MAX_SOURCE_LANG_RATIO = 0.35  # Max 35% source language remaining


class LanguageValidator:
    """Validates translation quality by checking language patterns."""
    
    def __init__(self):
        self.stats = {
            "total": 0,
            "passed": 0,
            "failed_source": 0,
            "failed_target": 0,
            "failed_unchanged": 0,
            "retried": 0,
        }
    
    def detect_language_ratio(self, text: str, lang_code: str) -> float:
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
        
        # For word-based patterns (like English)
        if lang_code == "en":
            total_words = len(text.split())
            return len(matches) / max(total_words, 1)
        
        # For character-based languages - count only text chars
        clean_text = re.sub(r'[\s\d\.,!?@#$%^&*()_+\-=\[\]{};\':"\\|<>\/~`「」『』【】。、！？]', '', text)
        total_chars = len(clean_text)
        
        return len(matches) / max(total_chars, 1)
    
    def detect_kana_ratio(self, text: str) -> float:
        """Specifically detect hiragana/katakana ratio (unique to Japanese)."""
        if not text:
            return 0.0
        
        kana_matches = re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text)
        clean_text = re.sub(r'[\s\d\.,!?@#$%^&*()_+\-=\[\]{};\':"\\|<>\/~`「」『』【】。、！？]', '', text)
        
        return len(kana_matches) / max(len(clean_text), 1)
    
    def validate(self, original: str, translated: str, source: str, target: str) -> Tuple[bool, str]:
        """
        Validate translation quality.
        
        Returns: (is_valid, reason)
        Reasons: "ok", "empty", "unchanged", "source_high", "target_low", "kana_high"
        """
        if not translated or not translated.strip():
            return False, "empty"
        
        # Identical = not translated
        if original.strip() == translated.strip():
            self.stats["failed_unchanged"] += 1
            return False, "unchanged"
        
        self.stats["total"] += 1
        
        # Japanese → Vietnamese: Check for remaining kana
        if source == "ja" and target == "vi":
            kana_ratio = self.detect_kana_ratio(translated)
            if kana_ratio > 0.15:
                self.stats["failed_source"] += 1
                return False, f"kana_high:{kana_ratio:.0%}"
        
        # Check source language ratio
        source_ratio = self.detect_language_ratio(translated, source)
        if source_ratio > MAX_SOURCE_LANG_RATIO:
            self.stats["failed_source"] += 1
            return False, f"source_high:{source_ratio:.0%}"
        
        # Check target language presence
        if target == "vi" and len(translated) > 15:
            target_ratio = self.detect_language_ratio(translated, target)
            if target_ratio < MIN_TARGET_LANG_RATIO:
                self.stats["failed_target"] += 1
                return False, f"target_low:{target_ratio:.0%}"
        
        self.stats["passed"] += 1
        return True, "ok"
    
    def validate_batch(self, originals: List[str], translations: List[str], 
                       source: str, target: str) -> Tuple[List[bool], List[str]]:
        """Validate batch of translations."""
        results = []
        reasons = []
        
        for orig, trans in zip(originals, translations):
            valid, reason = self.validate(orig, trans, source, target)
            results.append(valid)
            reasons.append(reason)
        
        return results, reasons
    
    def get_failure_rate(self, validations: List[bool]) -> float:
        """Calculate failure rate from validation results."""
        if not validations:
            return 0.0
        return validations.count(False) / len(validations) * 100
    
    def get_stats(self) -> Dict:
        """Get validation statistics."""
        total = self.stats["total"]
        if total == 0:
            return {"pass_rate": "N/A", **self.stats}
        
        pass_rate = self.stats["passed"] / total * 100
        return {
            "pass_rate": f"{pass_rate:.1f}%",
            **self.stats
        }
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


# Global validator instance for shared use
_validator = None

def get_validator() -> LanguageValidator:
    """Get or create global validator instance."""
    global _validator
    if _validator is None:
        _validator = LanguageValidator()
    return _validator
