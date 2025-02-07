from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the FAQ data
def load_faq_data():
    with open('data/faq_data.json') as file:
        faq_data = json.load(file)
    return faq_data

# Function to find the most similar question and return the answer
def find_similar_answer(user_query):
    faq_data = load_faq_data()

    # Extract questions and answers from the dataset
    all_questions = []
    answers = {}

    for entry in faq_data:
        questions = entry["questions"]
        answer = entry["answer"]
        # Add all questions to the list
        all_questions.extend(questions)
        # Map each question to its answer
        for question in questions:
            answers[question] = answer

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Combine the user query with all the questions for comparison
    combined_questions = all_questions + [user_query]

    # Vectorize the questions
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_questions)

    # Compute cosine similarity between the user query and all questions
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the index of the most similar question
    most_similar_index = cosine_sim.argmax()

    # Return the answer corresponding to the most similar question
    most_similar_question = all_questions[most_similar_index]
    return answers[most_similar_question]
