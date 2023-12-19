from django.shortcuts import render,redirect
from django.http import HttpResponse
from langchain.llms import HuggingFaceHub
import os
import shutil
from dotenv import load_dotenv
import requests
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from chromaviz import visualize_collection
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
# Create your views here.

# Creates Directory in base file for Vector store  
dir="vs"

# Load Environment Variables 
load_dotenv()
global embedding_function

# Function to Load URL as Document
def urlloader(website):
  url=[website]
  loader = UnstructuredURLLoader(urls=url)
  data = loader.load()
  return data

# Split URL to Chunks
def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks

# Store chunks into the vetorstore
def get_vectorstore(chunks,embedding_function):
    vectorstore = Chroma.from_documents(chunks, embedding_function,persist_directory=dir)
    vectorstore.persist()
    return vectorstore

# Load Already created vectorstore
def load_vectorstore(dir,embedding_function):
   persist_directory=dir
   vectorstore = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding_function)
   return vectorstore

# Langchain Qachain with the created vectorstore with Open AI
def conversation(vectorstore):
   llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k')
   qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
   return qa_chain

# Langchain Qachain with the created vectorstore with Google T5 Flan
def gconversation(vectorstore):
   llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature":0.5})
   qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
   return qa_chain

# Load Llama 2 7B LLM through Transformers Pipline
def llama2loadtransformer():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    global pipeline
    pipeline = pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    global llama2llm
    llama2llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.5})
    return llama2llm

# Langchain Qachain with the created vectorstore with Llama 2 7B
def llama2con(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm=llama2llm,chain_type="stuff",
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
    return qa_chain

# Load Flan Alpaca GPT 4 LLM through Transformers Pipline
def gpt4loadtransformer():
    model = "declare-lab/flan-alpaca-gpt4-xl"

    tokenizer = AutoTokenizer.from_pretrained(model)

    global pipeline
    pipeline = pipeline(
        "text2text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    global gpt4llm
    gpt4llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.5})
    return gpt4llm

# Langchain Qachain with the created vectorstore with Flan Alpaca GPT 4
def gpt4con(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm=gpt4llm,chain_type="stuff",
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
    return qa_chain

# Function to check if a file exists
def file_exists(file_name):
    return os.path.exists(file_name)

# Main Function Display Website
def home(request):
    # Defult To Display Start page
    if request.method == "GET":
        return render(request, "chatbot.html")
    
    # Uses User VectorStrore if Check Box is Checked
    elif request.method == "POST" and "checkvs" in request.POST : 
        # Takes User Input
        vs_input = request.POST.get("VSdir")
        embedding_function1 = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        global ulv
        ulv = load_vectorstore(vs_input,embedding_function1)
        return render(request, "chatbot.html",{"Loadedvs":"Used User vector store"})
    
    # If Process Button is Clicked
    elif request.method == "POST" and "process" in request.POST :
        # Uses Input for Website
        website = request.POST.get("Website_input")

        if website:
            # Loads information from Website and makes it into Chunks
            data = urlloader(website)
            txt_ch=get_text_chunks(data)

            # Option for different Text Embedding Function
                # All MiniLM L6 V2 Embedding
            if request.method == "POST" and "check1" in request.POST:
                embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                global gv
                gv = get_vectorstore(txt_ch,embedding_function)
                return render(request, "chatbot.html",{"Loadedvs":"Loaded Small/Fastest Embedding , Ask a Questions"})
                # Intfloat E5 Base Embedding
            elif request.method == "POST" and "check2" in request.POST:
                embedding_function = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-base")
                gv = get_vectorstore(txt_ch,embedding_function)
                return render(request, "chatbot.html",{"Loadedvs":"Loaded Medium/Fast Embedding , Ask a Questions"})
                # Intfloat E5 Large V2 Embedding
            elif request.method == "POST" and "check3" in request.POST:
                embedding_function = HuggingFaceInstructEmbeddings(model_name="intfloat/e5-large-v2")
                gv = get_vectorstore(txt_ch,embedding_function)
                return render(request, "chatbot.html",{"Loadedvs":"Loaded Large/Slow Embedding , Ask a Questions"})
        # If Website not Given Return Error
        else :
            return render(request, "chatbot.html",{"errorLoadedvs":"Enter Website URL"})

    # Button to delete Created Vector Store   
    elif request.method == "POST" and "del" in request.POST :
            if os.path.exists(dir):
                for file_name in os.listdir(dir):
                    # Try Except Block to handel Windows Errors
                    try:
                        file_path = os.path.join(dir, file_name)
                        os.remove(file_path)   
                        shutil.rmtree(dir)
                    except WindowsError as e:
                        if e.winerror == 3:
                            pass
   
                return render(request, "chatbot.html",{"Loadedvs":"Removed Existing Vector Store"})
            # Return Error if No Vector Store Created
            else:
                return render(request, "chatbot.html",{"Loadedvs":"No Current Vector Store Exists"})
            
    # Button to Load GPT-4 LLM (Should be used only ONCE while Website Running)        
    elif request.method == "POST" and "GPT4_load_btn" in request.POST :
        gpt4loadtransformer()
        return render(request, "chatbot.html",{"ConfMsg":"Loaded GPT-4 Locally"})
    
    # Button to Load Llama 2 LLM (Should be used only ONCE while Website Running)        
    elif request.method == "POST" and "llama_load_btn" in request.POST :
        llama2loadtransformer()
        return render(request, "chatbot.html",{"ConfMsg":"Loaded Llama 2 Locally"})
    
    # Button to display Results and References acoording to Which LLM is Picked 
    elif request.method == "POST" and "chat_btn" in request.POST :
        # User Input for Question
        user_input = request.POST.get("User_Input")
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # User Selected Model
        select_model = request.POST.get('LLMs')
        print(select_model)

        # If OpenAI selected from Drop down (Defult Option)
        if select_model == "OpenAI" or select_model == "" or select_model == None:
        
            if os.path.exists(dir):
                chat=conversation(gv) # QAchain with given Vectorstore
                result = chat(user_input) 
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
            # Use User Vector store
            elif 'ulv' in globals() :
                chat=conversation(ulv) # QAchain with given Vectorstore
                result = chat(user_input)
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given    
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
        # T5Flan selected from Drop down   
        elif select_model == "T5Flan":

            if os.path.exists(dir):
                chat=gconversation(gv) # QAchain with given Vectorstore
                # If User Question Exceeds Number of Tokens
                try:
                    result = chat(user_input)
                except ValueError:
                   return render(request, "chatbot.html",{"ErrorMsg":"Exceeded Number of Tokens, Ask another question"}) 
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
            # Use User Vector store
            elif 'ulv' in globals() :
                chat=gconversation(ulv) # QAchain with given Vectorstore
                try:
                    result = chat(user_input)
                except ValueError:
                   return render(request, "chatbot.html",{"ErrorMsg":"Exceeded Number of Tokens, Ask another question"}) 
                
                answer = result['result'] # Result
                # Get Reference and Websites
                try: 
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
        # Flan Alpaca GPT4 selected from Drop down     
        elif select_model == "FlanAlpacaGPT4":
            if os.path.exists(dir):
                chat=gpt4con(gv) # QAchain with given Vectorstore
                result = chat(user_input)
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
            # Use User Vector store
            elif 'ulv' in globals() :
                chat=gpt4con(ulv) # QAchain with given Vectorstore
                result = chat(user_input)
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
        # Llama2 selected from Drop down     
        elif select_model == "Llama2":
            if os.path.exists(dir):
                chat=llama2con(gv) # QAchain with given Vectorstore
                result = chat(user_input)
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
            # Use User Vector store
            elif 'ulv' in globals() :
                chat=llama2con(ulv) # QAchain with given Vectorstore
                result = chat(user_input)
                answer = result['result'] # Result
                # Get Reference and Websites
                try:
                    for Doc in result["source_documents"]:
                        ref=[]
                        ref.append(Doc.page_content)
                    for source in result["source_documents"]:
                        site=[]
                        site.append(source.metadata['source'])
                    separator = ' '
                    sources = separator.join(ref) # Reference
                    sites = separator.join(site) # Reference Websites
                # Display if No Reference given
                except UnboundLocalError:
                    return render(request, "chatbot.html",{"answer":answer,"sources":"No Sources","Websites":""})
                
                return render(request, "chatbot.html",{"answer":answer,"sources":sources,"Websites":sites})
            
    # Visualize Vectorstore          
    elif request.method == "POST" and "viz_btn" in request.POST:
        if os.path.exists(dir):
            vis=visualize_collection(gv._collection) # Visualize Vectorstore
            print(vis)
            return render(request, "chatbot.html",{"answer":answer,"sources":sources})
        
        # Use User Vector store
        elif 'ulv' in globals() :
            vis=visualize_collection(ulv._collection) #Visualize Vectorstore
            print(vis)
            return render(request, "chatbot.html",{"answer":answer,"sources":sources})
        
    # Close Visualisation
    elif request.method == "POST" and "clo_btn" in request.POST:
        return render(request, "chatbot.html")
    
    # If Vectore store Not created
    elif gv not in vars():
        return render(request, "chatbot.html",{"ErrorMsg":"Vector Store not Detected"})
    
    # If Vectore store Not created
    else:
        return render(request, "chatbot.html",{"ErrorMsg":"Enter Vector store"})



