import streamlit as st
import nltk
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('punkt')

# Load the saved Naive Bayes classifier
bayes = load('bayes_classifier.joblib')

# Load the vocabulary saved in Question 3 (must load BEFORE the function uses it)
with open('top_keys.txt', 'r') as f:
    topKeys = [word.strip() for word in f.readlines()]

# Sparse checklist: only send True words, absent words stay silent
def review_features_sparse(tokens):
    docSet = set(tokens)
    return {word: True for word in topKeys if word in docSet}

# Streamlit app
def main():
    st.title('Movie Review Sentiment Analysis')
    st.write('Enter a movie review to predict its sentiment.')

    review = st.text_area('Review:')

    if st.button('Predict Sentiment'):
        if review.strip() != '':
            # Same steps as the notebook loop: lowercase -> tokenize -> sparse -> predict
            tokens = nltk.word_tokenize(review.lower())
            features = review_features_sparse(tokens)
            dist = bayes.prob_classify(features)
            sentiment = dist.max()

            st.success(f'Predicted sentiment: {sentiment}')
            st.write(f'pos = {dist.prob("pos"):.3f}, neg = {dist.prob("neg"):.3f}')
        else:
            st.warning('Please enter a movie review.')

if __name__ == '__main__':
    main()