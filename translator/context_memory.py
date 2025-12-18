"""
Context Memory Module - OPTIMIZED for Coherence
Ensures consistent character voice, pronouns, and story flow across pages.

Key features:
- Character relationship tracking (who uses what pronouns with whom)
- Rolling context with recent dialogue
- Consistent term/name translation
- Story tone maintenance
"""
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, OrderedDict


class CharacterRelationship:
    """Tracks how one character addresses another."""
    
    def __init__(self, speaker: str, listener: str):
        self.speaker = speaker
        self.listener = listener
        self.pronoun_self = ""      # How speaker refers to self (tao, tôi, anh...)
        self.pronoun_other = ""     # How speaker refers to listener (mày, em, cậu...)
        self.honorific = ""         # Any honorific used (-san, tiền bối...)
        self.tone = "neutral"       # casual, formal, intimate, hostile
        self.examples: List[str] = []  # Example dialogues
    
    def update(self, self_pronoun: str = None, other_pronoun: str = None, 
               honorific: str = None, tone: str = None, example: str = None):
        if self_pronoun:
            self.pronoun_self = self_pronoun
        if other_pronoun:
            self.pronoun_other = other_pronoun
        if honorific:
            self.honorific = honorific
        if tone:
            self.tone = tone
        if example and len(self.examples) < 3:
            self.examples.append(example)
    
    def to_prompt(self) -> str:
        parts = [f"{self.speaker} → {self.listener}:"]
        if self.pronoun_self:
            parts.append(f"xưng '{self.pronoun_self}'")
        if self.pronoun_other:
            parts.append(f"gọi '{self.pronoun_other}'")
        if self.honorific:
            parts.append(f"kính ngữ '{self.honorific}'")
        if self.tone != "neutral":
            parts.append(f"giọng {self.tone}")
        return " ".join(parts)


class ContextMemory:
    """
    Advanced context memory for consistent manga translation.
    Tracks characters, relationships, and story flow.
    """
    
    # Pronoun patterns for detection
    PRONOUN_PATTERNS = {
        'vi': {
            'self': ['tao', 'tôi', 'tớ', 'mình', 'anh', 'chị', 'em', 'con', 'cháu', 'ta', 'bổn'],
            'other': ['mày', 'cậu', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà', 'ngươi', 'mi'],
        }
    }
    
    # Honorific patterns
    HONORIFIC_PATTERNS = [
        # Japanese
        (r'(\w+)[-\s]?(さん|san)', 'san'),
        (r'(\w+)[-\s]?(君|くん|kun)', 'kun'),
        (r'(\w+)[-\s]?(ちゃん|chan)', 'chan'),
        (r'(\w+)[-\s]?(様|さま|sama)', 'sama'),
        (r'(\w+)[-\s]?(先生|sensei)', 'sensei'),
        (r'(\w+)[-\s]?(先輩|senpai)', 'senpai'),
        # Korean
        (r'(\w+)[-\s]?(씨|ssi)', 'ssi'),
        (r'(\w+)[-\s]?(님|nim)', 'nim'),
        (r'(\w+)[-\s]?(선배|sunbae)', 'sunbae'),
        (r'(\w+)[-\s]?(오빠|oppa)', 'oppa'),
        (r'(\w+)[-\s]?(형|hyung)', 'hyung'),
        (r'(\w+)[-\s]?(누나|noona)', 'noona'),
        (r'(\w+)[-\s]?(언니|unnie)', 'unnie'),
        # Chinese
        (r'(\w+)[-\s]?(大人|daren)', 'daren'),
        (r'(\w+)[-\s]?(师父|shifu)', 'shifu'),
    ]
    
    # Name patterns
    NAME_PATTERNS = [
        r'\b([A-Z][a-z]{2,})\b',  # Capitalized names
        r'(\w{2,})[-\s]?(さん|君|ちゃん|様|先生|先輩)',  # Japanese with honorific
        r'(\w{2,})[-\s]?(씨|님|선배|오빠|형)',  # Korean with honorific
    ]
    
    # Config
    MAX_RECENT_PAGES = 8
    MAX_RECENT_DIALOGUES = 20
    MAX_CHARACTERS = 15
    MAX_TERMS = 30
    
    def __init__(self):
        """Initialize context memory."""
        self.reset()
    
    def reset(self):
        """Reset all context."""
        # Character tracking
        self.characters: Dict[str, Dict] = OrderedDict()  # {name: {info}}
        self.relationships: Dict[str, CharacterRelationship] = {}  # {speaker_listener: relationship}
        
        # Term consistency
        self.terms: Dict[str, str] = {}  # {original: translated}
        self.term_usage: Dict[str, int] = defaultdict(int)
        
        # Recent context
        self.recent_dialogues: List[Dict] = []  # [{original, translated, speaker?}]
        self.recent_pages: List[Dict] = []  # [{page, original, translated}]
        
        # Story tracking
        self.story_summary = ""
        self.current_scene = ""
        self.tone = "neutral"  # overall tone: action, comedy, romance, drama
    
    def add_character(self, name: str, info: Dict = None):
        """Add or update a character."""
        if name not in self.characters:
            self.characters[name] = {
                'name': name,
                'gender': 'unknown',
                'role': 'unknown',  # protagonist, antagonist, side
                'personality': '',
                'speech_style': '',  # formal, casual, rough, polite
                'first_seen': len(self.recent_pages),
            }
        
        if info:
            self.characters[name].update(info)
        
        # Limit characters
        while len(self.characters) > self.MAX_CHARACTERS:
            self.characters.popitem(last=False)
    
    def add_relationship(self, speaker: str, listener: str, **kwargs):
        """Add or update character relationship."""
        key = f"{speaker}_{listener}"
        
        if key not in self.relationships:
            self.relationships[key] = CharacterRelationship(speaker, listener)
        
        self.relationships[key].update(**kwargs)
    
    def detect_characters_from_text(self, original: str, translated: str):
        """Auto-detect character names and relationships from text."""
        # Detect names from original
        for pattern in self.NAME_PATTERNS:
            matches = re.findall(pattern, original)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                if len(name) >= 2:
                    self.add_character(name)
                    self.term_usage[name] += 1
        
        # Detect honorifics and relationships
        for pattern, honorific_type in self.HONORIFIC_PATTERNS:
            matches = re.findall(pattern, original)
            for match in matches:
                name = match[0] if isinstance(match, tuple) else match
                if name:
                    self.add_character(name)
                    # Store honorific usage
                    if name in self.characters:
                        self.characters[name]['honorific'] = honorific_type
    
    def detect_pronouns_from_translation(self, translated: str):
        """Detect pronoun patterns from Vietnamese translation."""
        # Self pronouns
        for pronoun in self.PRONOUN_PATTERNS['vi']['self']:
            if re.search(rf'\b{pronoun}\b', translated, re.IGNORECASE):
                self.term_usage[f"self:{pronoun}"] += 1
        
        # Other pronouns
        for pronoun in self.PRONOUN_PATTERNS['vi']['other']:
            if re.search(rf'\b{pronoun}\b', translated, re.IGNORECASE):
                self.term_usage[f"other:{pronoun}"] += 1
    
    def add_dialogue(self, original: str, translated: str, speaker: str = None):
        """Add a dialogue to recent context."""
        if not original or not translated:
            return
        
        self.recent_dialogues.append({
            'original': original[:200],  # Truncate long texts
            'translated': translated[:200],
            'speaker': speaker,
        })
        
        # Keep limited
        if len(self.recent_dialogues) > self.MAX_RECENT_DIALOGUES:
            self.recent_dialogues = self.recent_dialogues[-self.MAX_RECENT_DIALOGUES:]
        
        # Auto-detect from this dialogue
        self.detect_characters_from_text(original, translated)
        self.detect_pronouns_from_translation(translated)
    
    def update_from_batch(
        self, 
        original_texts: Dict[str, List[str]], 
        translated_texts: Dict[str, List[str]]
    ):
        """Update context after a batch translation."""
        for page_name in original_texts:
            orig_list = original_texts.get(page_name, [])
            trans_list = translated_texts.get(page_name, [])
            
            # Store page
            self.recent_pages.append({
                'page': page_name,
                'original': orig_list,
                'translated': trans_list,
            })
            
            # Process each dialogue
            for orig, trans in zip(orig_list, trans_list):
                self.add_dialogue(orig, trans)
        
        # Keep limited pages
        if len(self.recent_pages) > self.MAX_RECENT_PAGES:
            self.recent_pages = self.recent_pages[-self.MAX_RECENT_PAGES:]
        
        # Update summary
        self._update_summary()
    
    def _update_summary(self):
        """Update story summary from recent content."""
        if not self.recent_dialogues:
            self.story_summary = ""
            return
        
        # Get last few meaningful dialogues
        meaningful = [d['translated'] for d in self.recent_dialogues[-10:] 
                      if d['translated'] and len(d['translated']) > 15]
        
        if meaningful:
            self.story_summary = " → ".join(meaningful[-5:])
    
    def add_term(self, original: str, translated: str):
        """Add a term translation for consistency."""
        self.terms[original] = translated
        self.term_usage[original] += 5  # Boost priority
        
        # Limit terms
        if len(self.terms) > self.MAX_TERMS:
            # Remove least used
            sorted_terms = sorted(self.term_usage.items(), key=lambda x: x[1])
            for term, _ in sorted_terms[:10]:
                if term in self.terms:
                    del self.terms[term]
    
    def get_frequent_pronouns(self) -> Dict[str, List[str]]:
        """Get most frequently used pronouns."""
        self_pronouns = []
        other_pronouns = []
        
        for key, count in sorted(self.term_usage.items(), key=lambda x: -x[1]):
            if key.startswith('self:') and count >= 2:
                self_pronouns.append(key.replace('self:', ''))
            elif key.startswith('other:') and count >= 2:
                other_pronouns.append(key.replace('other:', ''))
        
        return {
            'self': self_pronouns[:5],
            'other': other_pronouns[:5],
        }
    
    def generate_context_prompt(self) -> str:
        """Generate context prompt for translator."""
        sections = []
        
        # 1. Character info (highest priority)
        if self.characters:
            char_lines = ["👤 NHÂN VẬT (giữ nhất quán):"]
            for name, info in list(self.characters.items())[:10]:
                line = f"  • {name}"
                if info.get('honorific'):
                    line += f" ({info['honorific']})"
                if info.get('speech_style'):
                    line += f" - {info['speech_style']}"
                char_lines.append(line)
            sections.append("\n".join(char_lines))
        
        # 2. Relationships and pronouns
        if self.relationships:
            rel_lines = ["🔗 QUAN HỆ XƯNG HÔ:"]
            for key, rel in list(self.relationships.items())[:8]:
                rel_lines.append(f"  • {rel.to_prompt()}")
            sections.append("\n".join(rel_lines))
        
        # 3. Frequent pronouns (guide consistency)
        pronouns = self.get_frequent_pronouns()
        if pronouns['self'] or pronouns['other']:
            pron_lines = ["📝 ĐẠI TỪ ĐÃ DÙNG (giữ nhất quán):"]
            if pronouns['self']:
                pron_lines.append(f"  • Xưng: {', '.join(pronouns['self'])}")
            if pronouns['other']:
                pron_lines.append(f"  • Gọi: {', '.join(pronouns['other'])}")
            sections.append("\n".join(pron_lines))
        
        # 4. Important terms
        if self.terms:
            term_lines = ["📖 THUẬT NGỮ:"]
            for orig, trans in list(self.terms.items())[:10]:
                term_lines.append(f"  • {orig} → {trans}")
            sections.append("\n".join(term_lines))
        
        # 5. Recent dialogue context (for flow)
        if self.recent_dialogues:
            recent = self.recent_dialogues[-6:]
            context_lines = ["💬 HỘI THOẠI GẦN ĐÂY (tham khảo giọng điệu):"]
            for d in recent:
                if d['translated']:
                    short = d['translated'][:80]
                    if len(d['translated']) > 80:
                        short += "..."
                    context_lines.append(f"  「{short}」")
            sections.append("\n".join(context_lines))
        
        if not sections:
            return ""
        
        return "═══ CONTEXT MEMORY ═══\n" + "\n\n".join(sections) + "\n═══════════════════════\n"
    
    def generate_consistency_rules(self) -> str:
        """Generate specific consistency rules based on observed patterns."""
        rules = []
        
        # Pronoun consistency rules
        pronouns = self.get_frequent_pronouns()
        if pronouns['self']:
            main_self = pronouns['self'][0]
            rules.append(f"• Nhân vật chính xưng '{main_self}' - GIỮ NGUYÊN xuyên suốt")
        
        if pronouns['other']:
            main_other = pronouns['other'][0]
            rules.append(f"• Cách gọi chính: '{main_other}' - GIỮ NGUYÊN")
        
        # Character-specific rules
        for name, info in list(self.characters.items())[:5]:
            if info.get('honorific'):
                hon = info['honorific']
                if hon in ['senpai', 'sunbae']:
                    rules.append(f"• {name} được gọi là 'tiền bối {name}' hoặc '{name} tiền bối'")
                elif hon in ['sensei', 'shifu']:
                    rules.append(f"• {name} được gọi là 'thầy {name}' hoặc '{name} thầy'")
                elif hon == 'sama':
                    rules.append(f"• {name} được gọi kính trọng: '{name}-sama' hoặc 'ngài {name}'")
        
        if rules:
            return "⚠️ QUY TẮC NHẤT QUÁN:\n" + "\n".join(rules)
        return ""
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            'characters': len(self.characters),
            'relationships': len(self.relationships),
            'terms': len(self.terms),
            'recent_dialogues': len(self.recent_dialogues),
            'recent_pages': len(self.recent_pages),
        }
    
    def clear(self):
        """Alias for reset."""
        self.reset()
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"ContextMemory(chars={stats['characters']}, "
                f"rels={stats['relationships']}, "
                f"dialogues={stats['recent_dialogues']})")
