from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
from database import cursor, db
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import fitz

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_PDFs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_dummy_answer(question):
    answer = "answer to this question : " + question
    return answer

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    load_dotenv()
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        pdf_text1 = extract_text_from_pdf1(file_path)

        # Store metadata and extracted text into the MySQL database
        upload_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO files (filename, upload_date) VALUES (%s, %s)", (filename, upload_date))
        db.commit()

        return jsonify({"success": True, "filename": filename})
    else:
        return jsonify({"success": False, "error": "Unsupported file format. Please upload a PDF file."})

def extract_text_from_pdf1(file_path):
    # Open the PDF file
    pdf_document = fitz.open(file_path)
    pdf_file_name = pdf_document.name[14:-4]
    print("\n uploaded pdf name : ",pdf_file_name)
    # Initialize an empty string to store text content
    text_content = ""
    # Iterate through each page of the PDF and extract text
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text_content += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000, # One paragraph size - total number of characters present in each chunk(para/section).
        chunk_overlap= 2000,
        length_function = len  
    )
    chunks = text_splitter.split_text(text=text_content)
    print("\n Chunks length : ", len(chunks))

    embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001")
    VectorStore = FAISS.from_texts(chunks,embedding = embeddings)
    print("\n VectoreStore : ", VectorStore)
    VectorStore.save_local("faiss_index")

def get_conversational_chain():
    prompt_template ="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not found in the context", don't provide the wrong answer \n
    Context :\n {context}?\n
    Question:\n{question}\n

    Answer:
    """  
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","questions"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents" :docs, "question": user_question},
        return_only_outputs=True)
    print("\n printing response : ",response)
    return response


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    print("\n Question came from front-end : ",question)
    response = user_input(question)
    print("\n answer to the question asked by an user : ", response)
    return jsonify({"success": True, "answer": response})

if __name__ == '__main__':
    app.run(debug=True)

