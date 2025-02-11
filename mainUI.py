import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import google.generativeai as genai
import dotenv
import re
import json

dotenv.load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "show_answers" not in st.session_state:
    st.session_state.show_answers = False

def parse_quiz_response(quiz_text):
    questions = []
    raw_questions = re.split(r'(?i)\nQuestion\s+\d+:', quiz_text)
    
    for q in raw_questions:
        if not q.strip():
            continue
            
        question = {}
        answer_match = re.search(r'(?i)(?:Answer|Correct\s*Answer)\s*:\s*([A-D])', q)
        
        if answer_match:
            answer = answer_match.group(1).upper()
            q_text = q[:answer_match.start()]
            
            options = {}
            option_matches = re.findall(r'(?i)^\s*([A-D])[\)\.]\s*(.+?)\s*$', q_text, flags=re.MULTILINE)
            
            if not option_matches:
                continue  
                
            for match in option_matches:
                options[match[0].upper()] = match[1].strip()
            
            question_text = re.sub(r'(?i)^\s*([A-D])[\)\.].+?$', '', q_text, flags=re.MULTILINE)
            question_text = re.sub(r'\s+', ' ', question_text).strip()
            
            if question_text and len(options) == 4:
                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct": answer
                })
    
    return questions or None

@st.cache_data
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = np.array(embedder.encode(chunks))
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    return text, chunks, index

def generate_quiz(text, num_questions=5, difficulty="medium"):
    prompt = f"""Generate {num_questions} multiple-choice questions from this text. Difficulty: {difficulty}.
    Format each question EXACTLY like this:
    
    Question X: [Your question here]
    A) Option 1
    B) Option 2
    C) Option 3
    D) Option 4
    Answer: [CORRECT_LETTER]

    Text: {text[:3000]}"""
    
    try:
        response = model.generate_content(prompt)
        parsed = parse_quiz_response(response.text)
        if not parsed:
            raise ValueError("No valid questions parsed")
        return parsed
    except Exception as e:
        st.error(f"Quiz generation failed: {str(e)}")
        return None

def calculate_score():
    score = 0
    for i, q in enumerate(st.session_state.quiz):
        if st.session_state.quiz_answers.get(str(i)) == q['correct']:
            score += 1
    return score

def download_quiz():
    quiz_data = ""
    for i, q in enumerate(st.session_state.quiz):
        quiz_data += f"Question {i+1}:\n{q['question']}\n"
        for letter, opt in q['options'].items():
            quiz_data += f"{letter}) {opt}\n"
        quiz_data += f"Correct Answer: {q['correct']}\n\n"
    return quiz_data

def main():
    st.set_page_config(page_title="PDF Insight AI", page_icon="📘", layout="wide")
    
    left_col, right_col = st.columns([1, 3])
    
    with right_col:
        st.header("Chat with your PDF")
        st.markdown("---")
        
        chat_container = st.container(height=500)
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about the document..."):
            if "index" not in st.session_state:
                st.error("Please upload a PDF document first!")
                return

        # Quiz Section
        if st.session_state.quiz:
            st.subheader("📝 Quiz Time!")
            progress = len(st.session_state.quiz_answers)/len(st.session_state.quiz)
            st.progress(progress)
            
            with st.expander("Quiz Controls"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Retry Quiz"):
                        st.session_state.quiz_answers = {}
                        st.session_state.show_answers = False
                with col2:
                    if st.button("📥 Download Quiz"):
                        quiz_file = download_quiz()
                        st.download_button(
                            label="Download Quiz",
                            data=quiz_file,
                            file_name="generated_quiz.txt",
                            mime="text/plain"
                        )
                with col3:
                    st.checkbox("Show Answers", 
                        key="show_answers_checkbox", 
                        value=st.session_state.show_answers,
                        on_change=lambda: setattr(st.session_state, 'show_answers', not st.session_state.show_answers))
            
            for i, question in enumerate(st.session_state.quiz):
                with st.container(border=True):
                    st.markdown(f"### Question {i+1}")
                    st.markdown(f"**{question['question']}**")
                    
                    answer_key = f"q_{i}"
                    options = list(question['options'].values())
                    
                    cols = st.columns(2)
                    for j, (letter, option) in enumerate(question['options'].items()):
                        with cols[j%2]:
                            if st.button(
                                f"{letter}) {option}",
                                key=f"opt_{i}_{j}",
                                use_container_width=True,
                                disabled=st.session_state.show_answers or answer_key in st.session_state.quiz_answers
                            ):
                                st.session_state.quiz_answers[str(i)] = letter
                                
                    if str(i) in st.session_state.quiz_answers:
                        user_answer = st.session_state.quiz_answers[str(i)]
                        correct = user_answer == question['correct']
                        
                        if correct:
                            st.success("Correct! 🎉")
                        else:
                            st.error(f"Wrong answer ❌. Correct: {question['correct']}) {question['options'][question['correct']]}")
                    
                    if st.session_state.show_answers:
                        st.info(f"Correct Answer: {question['correct']}) {question['options'][question['correct']]}")
            
            st.divider()
            score = calculate_score()
            st.metric("Your Score", f"{score}/{len(st.session_state.quiz)}", 
                     help="Your progress will be saved until you generate a new quiz")
            
    with left_col:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
        
        if pdf_file:
            text, chunks, index = process_pdf(pdf_file)
            st.session_state.text = text
            st.session_state.index = index
            st.success("PDF processed successfully!")
        
        # Add keys to maintain state
        num_questions = st.number_input("Number of Questions", 
                                  min_value=1, max_value=50, 
                                  value=5, step=1,
                                  key="num_questions")
        
        difficulty = st.select_slider("Select Difficulty",
                                options=["easy", "medium", "hard"],
                                value="medium",
                                key="difficulty")
        
        if st.button("Generate Quiz", type="primary"):
            if "text" not in st.session_state:
                st.error("Please upload and process a PDF first!")
            else:
                with st.spinner("Generating quiz questions..."):
                    quiz = generate_quiz(
                        st.session_state.text, 
                        num_questions, 
                        difficulty
                    )
                    if quiz:
                        st.session_state.quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.session_state.show_answers = False
                        st.success(f"Generated {len(quiz)} questions!")
                        st.rerun()  # Force UI refresh
                    else:
                        st.session_state.quiz = None

        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload a PDF document
        2. Ask questions in the chat
        3. Receive AI-powered answers
        4. Choose the number of quiz questions and difficulty level
        5. Click 'Generate Quiz' to create quiz questions with answers
        """)


if __name__ == "__main__":
    main()