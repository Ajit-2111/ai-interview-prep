import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)


def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):
    text = text.split()

    text = [y.lower() for y in text]

    return " ".join(text)


def Removing_punctuations(text):
    # Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )

    # remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()


def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    df.Text = df.Text.apply(lambda text: lower_case(text))
    df.Text = df.Text.apply(lambda text: remove_stop_words(text))
    df.Text = df.Text.apply(lambda text: Removing_numbers(text))
    df.Text = df.Text.apply(lambda text: Removing_punctuations(text))
    df.Text = df.Text.apply(lambda text: Removing_urls(text))
    df.Text = df.Text.apply(lambda text: lemmatization(text))
    return df


def normalized_sentence(text: str):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = Removing_numbers(text)
    text = Removing_punctuations(text)
    text = Removing_urls(text)
    text = lemmatization(text)
    return text


# loading
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load the saved label encoder object from disk
with open('./models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

model = load_model('./models/Emotion Recognition From English text.h5')


def textSentiment(sentence: str):
    print(sentence)
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba = np.max(model.predict(sentence))
    print(f"{result} : {proba}\n\n")
    return result


textSentiment("hello how are you ?")