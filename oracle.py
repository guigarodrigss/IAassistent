import streamlit as st
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

_ = load_dotenv(find_dotenv())

import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="167.86.120.109", 
        user="langchain_user",
        password="aaabbc123",
        database="copergas"
    )


ollama_server_url = "http://localhost:11434"
model_local = ChatOllama(model="llama3:latest")

@st.cache_resource
def load_api_data():
    url = "https://sgp.copergas.com.br/copergas-api-dev/clientes/getCliente.php?token=1e7ac7b8128880ea43363f56d3397595&segmento=3"
    
    # Ignorar verificação SSL
    response = requests.get(url, verify=False)  # Adicionando verify=False para ignorar a verificação SSL
    
    if response.status_code == 200:
        data = response.json()  # Supondo que a resposta da API seja no formato JSON
        
        # Imprimir a estrutura da resposta para diagnóstico
        st.write(data)  # Exibe o conteúdo de `data` no Streamlit
        
        # Verificar se a resposta é um dicionário
        if isinstance(data, dict):
            # A chave que contém o cliente está no próprio dicionário
            item = data  # Nesse caso, o item que estamos processando é o único que existe na resposta
            
            # Criar a pergunta e resposta baseadas nos dados do cliente
            question = f"Quais são os dados do cliente {item.get('nomeCliente', 'desconhecido')}?"
            answer = f"Nome: {item.get('nomeCliente', 'desconhecido')}, Razão Social: {item.get('razaoSocial', 'não disponível')}, CNPJ: {item.get('cnpj', 'não disponível')}, CEP: {item.get('cep', 'não disponível')}, Logradouro: {item.get('logradouro', 'não disponível')}, Bairro: {item.get('bairro', 'não disponível')}, Município: {item.get('municipio', 'não disponível')}, Segmento: {item.get('segmento', 'não disponível')}"
            
            documents = [{"question": question, "answer": answer}]
            
            embeddings = OllamaEmbeddings(base_url=ollama_server_url, model="nomic-embed-text:latest")
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            return retriever
        else:
            st.error("Estrutura de dados inesperada: Esperado um dicionário com os dados do cliente.")
            return None
    else:
        st.error(f"Erro ao acessar a API. Status: {response.status_code}")
        return None


retriever = load_api_data()

# Verificar se retriever foi corretamente gerado
if retriever is None:
    st.error("Erro ao carregar os dados da API. Não foi possível gerar o retriever.")
else:
    st.title("Assistente Virtual")


    rag_template = """
    Você é um atendente de uma empresa.
    Seu trabalho é conversar com os clientes, consultando a base de 
    conhecimentos da empresa, e dar 
    uma resposta simples e precisa para ele, baseada na 
    base de dados da empresa fornecida como 
    contexto.

    Contexto: {context}

    Pergunta do cliente: {question}
    """

    human = "{text}"
    prompt = ChatPromptTemplate.from_template(rag_template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model_local
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Você:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response_stream = chain.stream(user_input)
        
        full_response = ""
        
        response_container = st.chat_message("assistant")
        response_text = response_container.empty()
        
        for partial_response in response_stream:
            full_response += str(partial_response.content)
            response_text.markdown(full_response + "▌")

        st.session_state.messages.append({"role": "assistant", "content": full_response})