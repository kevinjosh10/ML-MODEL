"""
Tamil Character Vocabulary & Tokenizer for Speech-to-Text (ASR).
Handles Tamil Unicode graphemes, encoding to tensor indices, and CTC greedy decoding.
"""

from typing import List, Dict, Optional
try:
    import torch
except ImportError:
    torch = None
import numpy as np

# Base Tamil Unicode Characters and Modifiers
TAMIL_VOWELS = "அஆஇஈஉஊஎஏஐஒஓஔ"
TAMIL_CONSONANTS = "கஙசஞடணதநபமயரலவழளறன"
TAMIL_VOWEL_SIGNS = "ாிீுூெேைொோௌ்"
TAMIL_SPECIAL = "ஃ"
TAMIL_DIGITS = "௦௧௨௩௪௫௬௭௮௯"
PUNCTUATION_AND_SPACE = " .,!?'-\n"

# Special Tokens
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "

class TamilVocabulary:
    """
    Tamil Grapheme Vocabulary for Connectionist Temporal Classification (CTC) Speech Recognition.
    """
    def __init__(self):
        # CTC Blank Token is always at index 0
        self.special_tokens = [BLANK_TOKEN, UNK_TOKEN]
        
        # Build unique character list
        all_chars = [BLANK_TOKEN, UNK_TOKEN]
        
        # Add Tamil Unicode characters
        for char in TAMIL_VOWELS + TAMIL_CONSONANTS + TAMIL_VOWEL_SIGNS + TAMIL_SPECIAL + TAMIL_DIGITS:
            if char not in all_chars:
                all_chars.append(char)
                
        # Add basic punctuation and space
        for char in PUNCTUATION_AND_SPACE:
            if char not in all_chars:
                all_chars.append(char)
                
        # Also include standard ASCII letters and digits for code-mixed / English loan words
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            if char not in all_chars:
                all_chars.append(char)

        self.vocab: List[str] = all_chars
        self.char_to_id: Dict[str, int] = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.blank_id: int = self.char_to_id[BLANK_TOKEN]
        self.unk_id: int = self.char_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Converts a Tamil text string into a list of token IDs."""
        encoded = []
        for char in text:
            encoded.append(self.char_to_id.get(char, self.unk_id))
        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """Converts a list of token IDs directly to string without CTC blank collapsing."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_char and tid != self.blank_id:
                chars.append(self.id_to_char[tid])
        return "".join(chars)

    def ctc_greedy_decode(self, token_ids: List[int]) -> str:
        """
        Performs CTC Greedy Argmax decoding:
        1. Collapses consecutive duplicate tokens.
        2. Removes blank tokens (<blank>).
        """
        collapsed = []
        prev_token = None
        
        for tid in token_ids:
            if tid != prev_token:
                if tid != self.blank_id:
                    collapsed.append(self.id_to_char.get(tid, ""))
                prev_token = tid
                
        return "".join(collapsed).strip()

    def ctc_decode_batch(self, logits) -> List[str]:
        """
        Decodes a batch of CTC emission logits: (batch_size, time_steps, vocab_size).
        """
        if torch is not None and isinstance(logits, torch.Tensor):
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        elif isinstance(logits, np.ndarray):
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits
            
        results = []
        for seq in predictions:
            results.append(self.ctc_greedy_decode(seq.tolist() if hasattr(seq, 'tolist') else seq))
        return results

# Singleton vocabulary instance
tamil_vocab = TamilVocabulary()
