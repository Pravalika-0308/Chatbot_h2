# import os
# import time
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_community.chat_models import ChatOllama
# from config import GOOGLE_API_KEY
# from deep_translator import GoogleTranslator
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import google.generativeai as genai

# # Set the Streamlit page configuration and theme
# st.set_page_config(page_title="CHATBOT", layout="wide")
# st.header("AI-Based Interactive Chatbot for Law and Justice Services")

# # Sidebar configuration
# with st.sidebar:
#     st.title("C.H.A.T.B.O.T")
#     col1, col2, col3 = st.columns([1, 30, 1])
#     with col2:
#         st.image("images/Judge.png", use_column_width=True)
#     model_mode = st.toggle("Online Mode", value=True)
#     selected_language = st.selectbox("Start by Selecting your Language", 
#                                      ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", 
#                                       "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"])

# # Configure Google Generative AI
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-pro') 

# # Hide Streamlit's default menu
# def hide_hamburger_menu():
#     st.markdown("""
#         <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#         </style>
#         """, unsafe_allow_html=True)

# hide_hamburger_menu()

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "memory" not in st.session_state:
#     st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# # Function to split text into chunks
# @st.cache_resource
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create a vector store
# @st.cache_resource
# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store

# # Load and process your text data (Replace this with your actual legal text data)
# @st.cache_resource
# def load_legal_data():
#     text_data = """
#     [Your legal text data here]
#     """
#     chunks = get_text_chunks(text_data)
#     return get_vector_store(chunks)

# vector_store = load_legal_data()
# db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# def get_response_online(prompt, context):
#     full_prompt = f"""
#     As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
#     - Respond in a bullet-point format to clearly delineate distinct aspects of the legal query or service information.
#     - Each point should accurately reflect the breadth of the legal provision or service in question, avoiding over-specificity unless directly relevant to the user's query.
#     - Clarify the general applicability of the legal rules, sections, or services mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
#     - Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
#     - When asked about live streaming of court cases, provide the relevant links for court live streams.
#     - For queries about various DoJ services or information, provide accurate links and guidance.
#     - Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations or service information unless otherwise specified.
#     - Conclude with a brief summary that captures the essence of the legal discussion or service information and corrects any common misinterpretations related to the topic.

#     CONTEXT: {context}
#     QUESTION: {prompt}
#     ANSWER:
#     """
#     return model.generate_content(full_prompt, stream=True)

# def get_response_offline(prompt, context):
#     llm = ChatOllama(model="phi3")
#     # Implement offline response generation here
#     # This is a placeholder and needs to be implemented based on your offline requirements
#     return "Offline mode is not fully implemented yet."

# def translate_answer(answer, target_language):
#     translator = GoogleTranslator(source='auto', target=target_language)
#     translated_answer = translator.translate(answer)
#     return translated_answer

# def reset_conversation():
#     st.session_state.messages = []
#     st.session_state.memory.clear()

# def get_trimmed_chat_history():
#     max_history = 10
#     return st.session_state.messages[-max_history:]

# # Display messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Handle user input
# input_prompt = st.chat_input("Start with your legal query")
# if input_prompt:
#     st.session_state.messages.append({"role": "user", "content": input_prompt})
    
#     with st.chat_message("user"):
#         st.markdown(input_prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n"
        
#         # Retrieve context
#         context = db_retriever.get_relevant_documents(input_prompt)
#         context_text = "\n".join([doc.page_content for doc in context])
        
#         if model_mode:
#             response_stream = get_response_online(input_prompt, context_text)
#             for chunk in response_stream:
#                 full_response += chunk.text
#                 message_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.01)  # Adjust for smoother effect
#         else:
#             response = get_response_offline(input_prompt, context_text)
#             full_response += response
#             message_placeholder.markdown(full_response)

#         # Translate if necessary
#         if selected_language != "English":
#             translated_response = translate_answer(full_response, selected_language.lower())
#             message_placeholder.markdown(translated_response)
#         else:
#             message_placeholder.markdown(full_response)

#         st.session_state.messages.append({"role": "assistant", "content": full_response})

#     # Add a reset button after each interaction
#     if st.button('🗑️ Reset Conversation'):
#         reset_conversation()
#         st.experimental_rerun()

# # Footer
# def footer():
#     st.markdown("""
#         <style>
#         .footer {
#             position: fixed;
#             left: 0;
#             bottom: 0;
#             width: 100%;
#             background-color: #f1f1f1;
#             color: black;
#             text-align: center;
#         }
#         </style>
#         <div class="footer">
#         </div>
#         """, unsafe_allow_html=True)

# # Display the footer
# footer()




# import os
# import time
# import streamlit as st
# import speech_recognition as sr
# import pyttsx3
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_community.chat_models import ChatOllama
# from config import GOOGLE_API_KEY
# from deep_translator import GoogleTranslator
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import google.generativeai as genai

# # Set the Streamlit page configuration
# st.set_page_config(page_title="CHATBOT", layout="wide")
# st.header("AI-Based Interactive Chatbot for Law and Justice Services")

# # Initialize Text-to-Speech (TTS) engine
# tts_engine = pyttsx3.init()
# tts_engine.setProperty('rate', 150)  # Adjust voice speed

# # Sidebar settings
# with st.sidebar:
#     st.title("C.H.A.T.B.O.T")
#     col1, col2, col3 = st.columns([1, 30, 1])
#     with col2:
#         st.image("images/Judge.png", use_column_width=True)
#     model_mode = st.toggle("Online Mode", value=True)
#     selected_language = st.selectbox("Select your Language", 
#                                      ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", 
#                                       "Marathi", "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"])

# # Configure Google AI Model
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-pro')

# # Hide Streamlit menu
# def hide_hamburger_menu():
#     st.markdown("""
#         <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#         </style>
#         """, unsafe_allow_html=True)

# hide_hamburger_menu()

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "memory" not in st.session_state:
#     st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# # Voice Input Function
# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Listening... Speak now.")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             st.warning("Could not understand audio.")
#             return ""
#         except sr.RequestError:
#             st.error("Speech recognition service is unavailable.")
#             return ""

# # Text-to-Speech Output Function
# def speak_text(text):
#     tts_engine.say(text)
#     tts_engine.runAndWait()

# # Function to split text into chunks
# @st.cache_resource
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create a vector store
# @st.cache_resource
# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store

# # Load and process legal text data
# @st.cache_resource
# def load_legal_data():
#     text_data = """
#     [Your legal text data here]
#     """
#     chunks = get_text_chunks(text_data)
#     return get_vector_store(chunks)

# vector_store = load_legal_data()
# db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# def get_response_online(prompt, context):
#     full_prompt = f"""
#     As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, ensure responses are accurate and clear. 
#     - Provide legal information concisely.
#     - Offer official links when needed.
#     - Avoid assumptions and misinterpretations.
#     - Use bullet points for clarity.
    
#     CONTEXT: {context}
#     QUESTION: {prompt}
#     ANSWER:
#     """
#     return model.generate_content(full_prompt, stream=True)

# def get_response_offline(prompt, context):
#     llm = ChatOllama(model="phi3")
#     return "Offline mode is not fully implemented yet."

# def translate_answer(answer, target_language):
#     translator = GoogleTranslator(source='auto', target=target_language)
#     return translator.translate(answer)

# def reset_conversation():
#     st.session_state.messages = []
#     st.session_state.memory.clear()

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input options (Text & Voice)
# col1, col2 = st.columns([3, 1])

# with col1:
#     input_prompt = st.chat_input("Ask your legal question here")

# with col2:
#     if st.button("🎙️ Speak"):
#         input_prompt = recognize_speech()
#         if input_prompt:
#             st.success(f"Recognized: {input_prompt}")

# # Process user input
# if input_prompt:
#     st.session_state.messages.append({"role": "user", "content": input_prompt})

#     with st.chat_message("user"):
#         st.markdown(input_prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = "⚠️ **_Note: Ensure you verify legal advice._** \n\n"
        
#         # Retrieve relevant context
#         context = db_retriever.get_relevant_documents(input_prompt)
#         context_text = "\n".join([doc.page_content for doc in context])
        
#         if model_mode:
#             response_stream = get_response_online(input_prompt, context_text)
#             for chunk in response_stream:
#                 full_response += chunk.text
#                 message_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.01)
#         else:
#             full_response += get_response_offline(input_prompt, context_text)
#             message_placeholder.markdown(full_response)

#         # Translate if needed
#         if selected_language != "English":
#             translated_response = translate_answer(full_response, selected_language.lower())
#             message_placeholder.markdown(translated_response)
#             speak_text(translated_response)  # Speak translated response
#         else:
#             message_placeholder.markdown(full_response)
#             speak_text(full_response)  # Speak response

#         st.session_state.messages.append({"role": "assistant", "content": full_response})

#     # Reset button
#     if st.button('🗑️ Reset Conversation'):
#         reset_conversation()
#         st.experimental_rerun()

# # Footer
# def footer():
#     st.markdown("""
#         <style>
#         .footer {
#             position: fixed;
#             left: 0;
#             bottom: 0;
#             width: 100%;
#             background-color: #f1f1f1;
#             text-align: center;
#         }
#         </style>
#         <div class="footer">
#         </div>
#         """, unsafe_allow_html=True)

# footer()



import os
import json
import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import base64
from gtts import gTTS
from langchain.memory import ConversationBufferWindowMemory
from deep_translator import GoogleTranslator
from config import GOOGLE_API_KEY
import re
import unicodedata

# Mapping Streamlit language selection to gTTS language codes
LANGUAGE_MAP = {
    "English": "en", "Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Nepali": "ne", "Odia": "or", "Punjabi": "pa", "Sindhi": "sd",
    "Tamil": "ta", "Telugu": "te", "Urdu": "ur"
}

def clean_text_for_speech(text, language):
    """Removes only unwanted special characters while preserving non-English scripts."""
    if language == "English":
        return re.sub(r"[^\w\s.,!?]", "", text)  # Remove non-word characters for English
    else:
        return re.sub(r"[\*\-\^\$#@!~_+=\[\]{}()<>]", "", text)  # Remove specific symbols for other languages

def speak_text(text, language):
    """Convert text to speech and play it within Streamlit without external media player."""
    lang_code = LANGUAGE_MAP.get(language, "en")
    cleaned_text = clean_text_for_speech(text, language)
    
    try:
        tts = gTTS(text=cleaned_text, lang=lang_code)
        audio_file = "response.mp3"
        tts.save(audio_file)
        
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode()
            audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error in text-to-speech: {e}")

def load_user_data():
    if os.path.exists("user_data.json"):
        with open("user_data.json", "r") as f:
            return json.load(f)
    return {"history": [], "bookmarks": []}

def save_user_data(data):
    with open("user_data.json", "w") as f:
        json.dump(data, f, indent=4)

def get_response_online(prompt, context):
    full_prompt = f"""
    🤖 As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, provide accurate answers.
    🏛 CONTEXT: {context}
    ❓ QUESTION: {prompt}
    ✅ ANSWER:
    """
    return genai.GenerativeModel('gemini-1.5-pro').generate_content(full_prompt, stream=True)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    user_data["history"] = []
    save_user_data(user_data)
    st.experimental_rerun()

st.set_page_config(page_title="⚖️ CHATBOT", layout="wide")
st.header("🤖 AI-Based Interactive Chatbot for Law and Justice Services")

genai.configure(api_key=GOOGLE_API_KEY)
user_data = load_user_data()

with st.sidebar:
    st.title("🧑‍⚖️ C.H.A.T.B.O.T")
    st.image("images/Judge.png", use_column_width=True)
    model_mode = st.toggle("🌐 Online Mode", value=True)
    selected_language = st.selectbox("🌍 Select Language", list(LANGUAGE_MAP.keys()))
    
    if st.checkbox("📜 View Chat History"):
        st.write("### 📝 Chat History")
        for chat in user_data["history"]:
            st.write(f"{chat['role']}: {chat['content']}")
    
    if st.checkbox("🔖 View Bookmarks"):
        st.write("### 📌 Bookmarked Responses")
        for bookmark in user_data["bookmarks"]:
            st.write(f"📍 {bookmark}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

col1, col2 = st.columns([5, 1])
with col1:
    input_prompt = st.chat_input("💬 Start with your legal query")

with col2:
    if st.button("🎤 Speak"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            try:
                language_code = LANGUAGE_MAP.get(selected_language, "en")
                audio = recognizer.listen(source, timeout=5)
                input_prompt = recognizer.recognize_google(audio, language=language_code)
            except sr.UnknownValueError:
                input_prompt = "⚠️ Sorry, I couldn't understand. Try again."
            except sr.RequestError:
                input_prompt = "❌ Error in recognizing speech."

if input_prompt:
    if selected_language != "English":
        input_prompt = GoogleTranslator(source=selected_language.lower(), target="en").translate(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    user_data["history"].append({"role": "user", "content": input_prompt})
    save_user_data(user_data)
    
    with st.chat_message("user"):
        st.markdown(f"👤 {input_prompt}")
    
    with st.chat_message("assistant"):
        context = ""
        response_stream = get_response_online(input_prompt, context) if model_mode else "⚡ Offline mode is not fully implemented yet."
        
        full_response = "".join(chunk.text for chunk in response_stream)
        
        if selected_language != "English":
            full_response = GoogleTranslator(source="en", target=selected_language.lower()).translate(full_response)
        
        st.write(f"🤖 {full_response}")
        speak_text(full_response, selected_language)
        
        user_data["history"].append({"role": "assistant", "content": full_response})
        save_user_data(user_data)
        
        if st.button("🔖 Bookmark Response"):
            user_data["bookmarks"].append(full_response)
            save_user_data(user_data)
            st.success("✅ Response Bookmarked!")
    
    if st.button("🗑️ Reset Conversation"):
        reset_conversation()
