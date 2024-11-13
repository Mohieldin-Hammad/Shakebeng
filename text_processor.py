import os
import requests
import random
import torch
import logging
import re
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
from pathlib import Path
from unicodedata import normalize, category


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextProcessor:

    SPECIAL_TOKENS =  ['<UNK>', '<START>', '<END>']

    def __init__(
        self,
        min_word_freq: int = 2,
        min_word_length: int = 1,
        max_word_length: int = 50,
        lowercase: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        custom_filters: Optional[List[Tuple[str, str]]] = None):

        self.min_word_freq = min_word_freq
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode
        self.custom_filters = custom_filters or []

        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.number_pattern = re.compile(r'\d+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.character_pattern = re.compile(r'^[A-Z][A-Za-z]+\s?[A-Za-z]*:$')
        
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.stop_words: Set[str] = set()
        

    def load_stop_words(self, file_path: Optional[str] = None) -> None:
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.stop_words = set(line.strip() for line in f)
                    logger.info(f"Loaded {len(self.stop_words)} stop words from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load stop words: {e}")

        else:
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                'that', 'the', 'to', 'was', 'were', 'will', 'with'
            }
    
    def load_text(self, file_path: str, fallback_url: Optional[str] = None) -> str:
        file_path = os.path.join(os.getcwd(), file_path)
        if not os.path.exists(file_path):
            if not fallback_url:
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.info(f"Downloading text from {fallback_url}")
            try:
                response = requests.get(fallback_url, timeout=10)
                response.raise_for_status()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            except requests.RequestException as e:
                logger.error(f"Failed to download text: {e}")
                raise
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Loaded text with length: {len(text)}")
            return text
        except UnicodeDecodeError as e:
            logger.error("Failed to decode text file.")
            raise
                
    def tag_speakers(self, text: str) -> str:
        lines = text.strip().split('\n')
        tagged_lines = []
        for line in lines:
            if self.character_pattern.match(line):
                tagged_lines.append(f'<SPEAKER> {line}')
            else:
                tagged_lines.append(line)
        return '\n'.join(tagged_lines)
                

    def clean_text(self, text: str) -> str:
        if not text:
            return text
        
        if self.normalize_unicode:
            text = normalize('NFKD', text)

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)

        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)

        if self.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
        
        for pattern, replacement in self.custom_filters:
            text = re.sub(pattern, replacement, text)

        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text

    def clean_paragraph(self, text: str) -> List[str]:
        placeholder_token = "temp_placeholder_token"
        text = self.tag_speakers(text).replace("<SPEAKER>", placeholder_token)
        
        text = self.clean_text(text).replace(placeholder_token, '<SPEAKER>').strip().split()
        return [
            word for word in text
            if (
                (self.min_word_length <= len(word) <= self.max_word_length)
                and word not in self.stop_words
                and (not self.remove_numbers or not word.isdigit())
            )
        ]
        
 

    def clean_paragraphs(self, paragraphs: List[str]) -> List[List[str]]:
        cleaned_paragraphs = [self.clean_paragraph(para) for para in paragraphs if para]
        logger.info(f"Cleaned {len(paragraphs)} paragraphs")
        return cleaned_paragraphs

    def build_paragraphs(self, text: str) -> List[str]:
        paragraphs = [para.strip() for para in text.strip().split("\n\n") if para.strip()]
        logger.info(f"Built {len(paragraphs)} paragraphs")
        return paragraphs

    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        words = text.strip().split()
        word_counts = Counter(words)
        freq_words = {word for word, count in word_counts.items() if count >= self.min_word_freq}
        vocab = self.SPECIAL_TOKENS + sorted(freq_words)
        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        
        logger.info(f"Built vocabulary with {len(vocab)} words")
        logger.debug(f"Vocabulary reduction: {len(word_counts)} -> {len(vocab)} words")
        
        return self.word2idx, self.idx2word

    def process_text(self, text: str) -> Tuple[str, List[List[str]]]:
        paragraphs = self.build_paragraphs(text)
        cleaned_paragraphs = self.clean_paragraphs(paragraphs)
        cleaned_text = "\n".join(" ".join(para) for para in cleaned_paragraphs)
        return cleaned_text, cleaned_paragraphs


    def build_dataset(
            self,
            paragraphs: List[List[str]],
            block_size=3,
            max_examples_per_paragraph: Optional[int] = None,
            device: Optional[torch.device]=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if block_size < 1:
            raise ValueError("block_size must be greater than 0")
        if not self.word2idx:
            raise ValueError("Vocabulary not built. Run build_vocab() first.")
        
        X, Y = [], []
        unk_idx = self.word2idx['<UNK>']
        start_idx = self.word2idx['<START>']
        end_idx = self.word2idx['<END>']

        for para in paragraphs:
            # if max_examples_per_paragraph:
            #     para = para[:max_examples_per_paragraph]
            para = ["<START>"] + para + ["<END>"]
            para_expamples = []
            for i in range(len(para) - block_size):
                context = para[i:i+block_size]
                target = para[i+block_size]

                context_idxs = self.word_to_index(context) #[self.word2idx.get(word, unk_idx) for word in context]
                target_idx = self.word2idx.get(target, unk_idx)
                para_expamples.append((context_idxs, target_idx))

            if max_examples_per_paragraph and len(para_expamples) > max_examples_per_paragraph:
                para_expamples = random.sample(para_expamples, max_examples_per_paragraph)
            
            X.extend(ex[0] for ex in para_expamples)
            Y.extend(ex[1] for ex in para_expamples)
        
        X_tensor = torch.tensor(X, dtype=torch.long)
        Y_tensor = torch.tensor(Y, dtype=torch.long)
        if device:
            X_tensor = X_tensor.to(device)
            Y_tensor = Y_tensor.to(device)
        logger.info(f"Built dataset with {len(X_tensor)} examples")
        return X_tensor, Y_tensor


    def word_to_index(self, word: List[str]) -> List[int]:
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in word]