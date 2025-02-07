import json
from nlp_model.similarity import get_best_answer

def load_faq_data():
    # Load FAQ dataset from a JSON file
    with open('data/faq_data.json', 'r') as file:
        faq_data = json.load(file)
    return faq_data

def respond_to_query(query):
    # Load FAQ data
    faq_data = load_faq_data()
    
    # Get the best matching answer from the FAQ dataset
    return get_best_answer(query, faq_data)
