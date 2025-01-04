import os
import json
import datetime
import csv
import nltk
import ssl
import pandas
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# Title of the app
st.title('🤖 Intents Based Chatbot using NLPt')
st.write("Ask me anything and I'll do my best to help!")

# Initialize chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Create a sidebar menu with options
menu = ['Home', 'Conversation History', 'About']
choice = st.sidebar.selectbox('Menu', menu)

# Home Menu
if choice == 'Home':

    # Check if the chat log file exists, and if not, create it with column names
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    # Text input with callback to reset the field
    def on_user_input_change():
        # This will be triggered when user presses Enter or sends the message
        if st.session_state['user_input']:
            # Append user input to chat history
            st.session_state['chat_history'].append(('You', st.session_state['user_input']))

            # Get the chatbot response
            response = chatbot(st.session_state['user_input'])

            # Append bot response to chat history
            st.session_state['chat_history'].append(('Bot', response))

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([st.session_state['user_input'], response, timestamp])

            # Clear the input field after processing    # This resets the input field
            st.session_state['user_input'] = ""  

    # User input field with on_change callback
    st.text_input("You:", key='user_input', on_change=on_user_input_change)

    # Display chat history
    for sender, message in reversed(st.session_state['chat_history']):
        alignment = 'right' if sender == "You" else 'left'
        st.markdown(
            f"<div style='text-align:{alignment};'><b>{sender}:</b> {message}</div>",
            unsafe_allow_html=True
        )


# Conversation History Menu
elif choice == "Conversation History":
    st.header("📚 Conversation History")
    
    if os.path.exists('chat_log.csv'):
        # Read the chat history using Pandas
        df = pd.read_csv('chat_log.csv')
        
        # Check if the CSV is empty
        if df.empty:
            st.warning("No conversation history available.")
        else:
            # Display the chat history in a dataframe
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No conversation history found. Start a chat to save interactions.")


elif choice == "About":
    st.write("The chatbot project demonstrates the integration of Natural Language Processing (NLP) with a user-friendly interface to simulate human-like conversation. It leverages machine learning techniques to understand and respond to user queries.")

    st.subheader("Project Overview:")
    st.write("""
    This project focuses on developing an AI-driven chatbot capable of interpreting user intent and delivering relevant responses. The system uses NLP techniques for text preprocessing and machine learning algorithms for classification.
    """)

    st.subheader("Dataset:")
    st.write("""
    The dataset comprises labeled text inputs categorized into predefined intents. Each intent is associated with multiple patterns (sample user inputs) and corresponding responses.
    """)

    st.subheader("Streamlit Chatbot Interface:")
    st.write("The chatbot interface is built with Streamlit, allowing seamless interaction between users and the model. The intuitive design ensures ease of use while displaying real-time chatbot responses.")

    st.subheader("Conclusion:")
    st.write("This project highlights the potential of NLP and machine learning in building intelligent chatbot systems. The combination of Streamlit for the interface and Logistic Regression for intent classification ensures scalability and reliability.")

# Footer
st.markdown("---")
st.markdown("### 📚 Project by Priyanka Kumari")
st.write("© 2024 Priyanka Kumari. All Rights Reserved.")

