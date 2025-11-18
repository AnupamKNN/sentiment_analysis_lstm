"""
Text preprocessing utilities
"""

import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from tqdm import tqdm

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass


class TextPreprocessor:
    """Text preprocessing operations"""
    
    @staticmethod
    def clean_social_media_text(text: str) -> str:
        """Clean social media text"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Tokenize text"""
        if pd.isna(text) or text == '':
            return []
        
        try:
            return word_tokenize(str(text))
        except:
            return str(text).split()
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Clean dataframe"""
        tqdm.pandas(desc="Cleaning")
        df['text_cleaned'] = df[text_column].progress_apply(
            TextPreprocessor.clean_social_media_text
        )
        
        # Handle empty texts
        empty_mask = df['text_cleaned'].str.len() == 0
        if empty_mask.sum() > 0:
            df.loc[empty_mask, 'text_cleaned'] = df.loc[empty_mask, text_column].str.lower()
        
        return df
    
    @staticmethod
    def tokenize_dataframe(df: pd.DataFrame, 
                          text_column: str = 'text_cleaned',
                          chunk_size: int = 100000) -> pd.DataFrame:
        """Tokenize dataframe in chunks"""
        total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        tokenized_results = []
        
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(df))
            
            print(f"Chunk {chunk_idx + 1}/{total_chunks}")
            
            chunk_tokens = df.iloc[start:end][text_column].apply(
                TextPreprocessor.tokenize_text
            )
            tokenized_results.extend(chunk_tokens.tolist())
        
        df['tokens'] = tokenized_results
        df['token_count'] = df['tokens'].apply(len)
        df['tokens_str'] = df['tokens'].apply(lambda x: '|'.join(x) if isinstance(x, list) else '')
        
        return df
