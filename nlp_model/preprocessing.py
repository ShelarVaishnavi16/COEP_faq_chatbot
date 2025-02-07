import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(user_input):
    # Tokenize the input text
    tokens = word_tokenize(user_input)
    
    # Remove stopwords and lemmatize the tokens
    filtered_tokens = [
        lemmatizer.lemmatize(word.lower()) 
        for word in tokens if word.lower() not in stop_words
    ]
    
    return " ".join(filtered_tokens)
