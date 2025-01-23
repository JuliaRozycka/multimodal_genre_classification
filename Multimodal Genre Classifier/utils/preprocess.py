import string
import pandas as pd
from tqdm import tqdm
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class LyricPreprocessor:
    def __init__(self, lyrics: str):
        self.lyrics = lyrics
        
    def preprocess_for_nlp(self, data: pd.DataFrame) -> pd.DataFrame:
        punctuation_to_remove = string.punctuation.replace("'", "")
        data[self.lyrics] = data[self.lyrics].str.lower()
        data[self.lyrics] = data[self.lyrics].str.replace('chorus', '')
        data[self.lyrics] = data[self.lyrics].str.replace('verse', '')
        data[self.lyrics] = data[self.lyrics].str.replace(f"[{punctuation_to_remove}]", "", regex=True)
        data[self.lyrics] = data[self.lyrics].str.replace('2x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x2', '')
        data[self.lyrics] = data[self.lyrics].str.replace('3x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x3', '')
        data[self.lyrics] = data[self.lyrics].str.replace('4x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x4', '')
        data[self.lyrics] = data[self.lyrics].str.replace('5x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x5', '')
        data[self.lyrics] = data[self.lyrics].str.replace('6x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x6', '')
        data[self.lyrics] = data[self.lyrics].str.replace('7x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x7', '')
        data[self.lyrics] = data[self.lyrics].str.replace('8x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x8', '')
        data[self.lyrics] = data[self.lyrics].str.replace('9x', '')
        data[self.lyrics] = data[self.lyrics].str.replace('x9', '')
        data[self.lyrics] = data[self.lyrics].str.replace('\n', ' ')
        data[self.lyrics] = data[self.lyrics].str.replace(r'\s+', ' ', regex=True).str.strip()
        data[self.lyrics] = data[self.lyrics].str.replace(r'\d+', '', regex=True)
        data[self.lyrics] = data[self.lyrics].str.replace("'", "")
        return data

    def handle_contractions(self, df: pd.DataFrame) -> pd.DataFrame:
        tqdm.pandas(desc="Handling contractions")
        data = df.copy()
        data[self.lyrics] = data[self.lyrics].progress_apply(lambda x: contractions.fix(x))
        return data

    def lemmatize_lyrics(self, df: pd.DataFrame, output_column: str) -> pd.DataFrame:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()

        # Function to lemmatize lyrics
        def lemmatize_lyrics(lyrics):
            tokens = word_tokenize(lyrics)
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)

        # Create sets
        tqdm.pandas()
        df[output_column] = df[self.lyrics].progress_apply(lemmatize_lyrics)
        return df

    def preprocess_lyrics(self, data: pd.DataFrame, output_column: str) -> pd.DataFrame:
        data = self.preprocess_for_nlp(data)
        data = self.handle_contractions(data)
        data = self.lemmatize_lyrics(data, output_column)
        return data