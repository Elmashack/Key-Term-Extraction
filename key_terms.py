import string
import nltk

from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

NOUN_TAG = 'NN'

file_path = 'news.xml'
dataset = []
lemming = WordNetLemmatizer()
stop_words = set(stopwords.words('english') + list(string.punctuation))
root = etree.parse(file_path).getroot()
titles = []
# Text preprocessing
for news in root[0]:
    titles.append(news.find(".//*[@name='head']").text)
    story = news.find(".//*[@name='text']")
    tokens = word_tokenize(story.text.lower())
    # get rid of stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # lemmatization
    tokens = [lemming.lemmatize(token.lower()) for token in tokens]
    # get nouns only
    tokens = [word for word in tokens if nltk.pos_tag([word])[0][1] in NOUN_TAG]
    dataset.append(' '.join(tokens))

vectorizer = TfidfVectorizer(input='content', ngram_range=(1, 1))
matrix = vectorizer.fit_transform(dataset)
terms = vectorizer.get_feature_names_out()
for row in range(len(matrix.toarray())):
    freq_word = {}
    for i in range(len(matrix.toarray()[row])):
        freq_word[terms[i]] = matrix.toarray()[row][i]
    print(titles[row] + ':')
    [print(key, end=' ') for key, val in sorted(freq_word.items(), key=lambda item: (item[1], item[0]), reverse=True)[:5]]
    print('\n')

