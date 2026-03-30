import os
import pickle
import subprocess
from req_res import Request, Response
#import google.generativeai as genai
from google import genai
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# translation_model = None
# translation_tokenizer = None

# def load_translation_model():
#     global translation_model, translation_tokenizer

#     if translation_model is None:
#         from transformers import MarianMTModel, MarianTokenizer

#         model_name = "Helsinki-NLP/opus-mt-en-hi"

#         translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
#         translation_model = MarianMTModel.from_pretrained(model_name)

#     return translation_tokenizer, translation_model 
# ##



def init_llm_model(api_key=None):
    if api_key is None:
            raise ValueError("API Key is required")

    client = genai.Client(api_key=api_key)
    return client

def embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",#"all-mpnet-base-v2"
        model_kwargs={"device": "cpu"}
    )
  
# def eng_hindi(text):
#     tokenizer, model = load_translation_model()
#     tokens = tokenizer(text, return_tensors="pt", padding=True)                 
#     translated = model.generate(**tokens)
#     output = tokenizer.decode(translated[0], skip_special_tokens=True)

#     return output

def load_index(filename, force_rebuild_index=False):
    if force_rebuild_index or not os.path.exists(filename):
        print("Force Rebuilding Index...")
        cmd = "python build_index.py"
        subprocess.call(cmd, shell=True)

    with open(filename, "rb") as f:
        vector_store = pickle.load(f)

    return vector_store


#Hypothetical Document Embedding (HyDE) in Document Retrieval


class HyDERetriever:
    def __init__(self, file_path, chunk_size=500, chunk_overlap=100):
        self.llm =ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

        self.embeddings = embedding_model()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(file_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    
        
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vector_store.similarity_search_with_relevance_scores(hypothetical_doc, k=5)
        return similar_docs, hypothetical_doc