# Code for Master Thesis ...

# Components

1. Audio analysis -song_audio directory
2. Lyrics analysis - song_lyrics directory
3. Metadata analysis - song_metadata directory
4. Multimodal analysis - Multimodal Genre Classifier

Requirements are in the requirements.txt file. Used Python: 3.12.0.

# Important classes

### Multimodal Genre Classifier -> utils -> melspectrograms_librosa.py

> The provided code contains three functions. The `mp3_to_mel_spectrogram` function converts an MP3 file to a mel
> spectrogram, optionally cutting it to a fixed length. The `save_mel_spectrogram` function saves a given mel spectrogram
> to a specified file path. The `mel_spectrogram` function combines the previous two functions by converting an MP3 file
> to a mel spectrogram and then saving it to a file.

### Multimodal Genre Classifier -> utils -> preprocessing.py

> The `LyricPreprocessor` class in `preprocess.py` provides methods to preprocess song lyrics for natural language
> processing (NLP) tasks. It includes functions to clean and normalize text by removing punctuation, specific keywords,
> and numerical repetitions, as well as handling contractions. The class also offers methods for lemmatizing and stemming
> words, and removing stop words. The `preprocess_lyrics` method combines these preprocessing steps into a single
> function, making it easy to prepare lyrics data for further analysis or machine learning tasks.

# Audio analysis

The audio analysis is done using the Librosa library. The audio features are extracted from the audio files and saved
in a csv file. The audio features are: chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff,
zero_crossing_rate, mfcc1-13, rmse and tempo.

### Files content

- cnn_mel_spectograms.ipynb - Convolutional Neural Network for Mel Spectograms
- resnet_mel_spectograms.ipynb - ResNet for Mel Spectograms
- resnet_rnn_hybrid.ipynb - ResNet and RNN Hybrid Model
- extract_featurs_from_genres.ipynb - Extracting features from genres

### Experiments

Used dataset was MTG-Jamendo dataset. The mp3 files were downloaded and features and mel spectograms were extracted.
Then the files were used in 4 different models: XGBoost, CNN for Mel Spectograms, ResNet for Mel Spectograms,
ResNet and RNN Hybrid Model. The models were compared with F1-score and the best model was ResNet for Mel Spectograms.

# Lyrics analysis

Lyrics are taken from 3 kaggle datasets combined and 8 genres were selected. The lyrics are preprocessed to 2
models: BiLSTM-CNN hybrid and RoBERTa. The models are compared with F1-score and the best model was RoBERTa.

### Files content

- NLP_BERT.ipynb - RoBERTa model
- NLP_GloVe_LSTM.ipynb - BiLSTM-CNN hybrid model

# Metadata analysis

The metadata analysis is done using the XGBoost and Random Forest. The metadata features were taken from 3 datasets (
this data was probably taken from Spotify API), the numerical features were taken. Best model was XGBoost.

### Files content

- gather_dataset_rf_xgboost.ipynb - Gathering dataset and training Random Forest and XGBoost models

# Multimodal analysis

The multimodal analysis is done using the Multimodal Genre Classifier. The audio features, lyrics and metadata features
are combined and used in the model. The model is a fusion model with 3 modalities: audio, lyrics and metadata. The model
is compared with F1-score and the best model was the early fusion. The model is trained on created with FMA and lyrics
dataset.

### Files content

#### Unimodal models

The unimodal models were tested on the new dataset in files audio.ipynb, lyrics.ipynb and metadata.ipynb.
Two fusions were tested in: LOOCV_XGBoost.ipynb (early fusion) and weighted_voting.ipynb (late fusion).
