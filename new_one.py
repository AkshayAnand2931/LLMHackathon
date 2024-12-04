import os
import sqlite3
import json
from typing import Literal, List, Dict, Any
from datetime import datetime
import requests
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
class ChatHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history"""
        self.messages.append({"role": role, "content": content, "timestamp": datetime.now()})
    
    def get_messages(self, limit: int = None) -> List[Dict]:
        """Get recent messages from history"""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear(self):
        """Clear chat history"""
        self.messages = []

class ChromaDBTool:
    def __init__(self, data_path: str, lmstudio_api_base: str = "http://localhost:1234/v1"):
        self.data_path = data_path
        self.lmstudio_api_base = lmstudio_api_base
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = ChatHistory()
    
    def load_existing_vector_store(self):
        """Load or create ChromaDB vector store"""
        try:
            # Initialize ChromaDB with persistent directory
            self.vector_store = Chroma(
                persist_directory=os.path.join(self.data_path, "chroma_db"),
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = None

    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        try:
            if not self.vector_store:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=os.path.join(self.data_path, "chroma_db")
                )
            else:
                self.vector_store.add_documents(documents)
            self.vector_store.persist()
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def retrieve(self, query: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve documents from vector store based on query"""
        try:
            if not self.vector_store:
                self.load_existing_vector_store()
            
            if query:
                # Add query to chat history
                self.chat_history.add_message("user", query)
                
                results = self.vector_store.similarity_search_with_relevance_scores(query, k=limit)
                retrieved_docs = [
                    {
                        "content": doc[0].page_content,
                        "metadata": doc[0].metadata,
                        "score": doc[1]
                    }
                    for doc in results
                ]
                
                # Add response to chat history
                self.chat_history.add_message("assistant", f"Retrieved {len(retrieved_docs)} documents")
                
                return retrieved_docs
            return []
            
        except Exception as e:
            print(f"Error retrieving from vector store: {e}")
            return []

    def query(self, question: str) -> str:
        """Query the RAG system with chat history context"""
        try:
            if not self.qa_chain:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                
                llm = OpenAI(
                    base_url=self.lmstudio_api_base,
                    api_key="not-needed", 
                    streaming=True,
                    callback_manager=callback_manager,
                    temperature=0.7,
                )

                prompt_template = """Use the following pieces of context and chat history to answer the question at the end. 
                If you don't know the answer, just say that you don't know.

                Context: {context}

                Chat History:
                {chat_history}

                Question: {question}

                Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "chat_history", "question"]
                )

                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 3}
                    ),
                    chain_type_kwargs={"prompt": PROMPT}
                )
            
            # Get recent chat history
            recent_history = self.chat_history.get_messages(limit=5)
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            
            # Add question to chat history
            self.chat_history.add_message("user", question)
            
            response = self.qa_chain.run(
                question=question,
                chat_history=history_str
            )
            
            # Add response to chat history
            self.chat_history.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return f"Error: {str(e)}"

class WebSearchTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform web search using SerpAPI"""
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" in results:
                return [
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    }
                    for item in results["organic_results"][:num_results]
                ]
            return []
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []

def display_image(image_path: str = None, image_url: str = None):
    """Display image from local path or URL"""
    try:
        if image_path:
            img = Image.open(image_path)
        elif image_url:
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
        else:
            raise ValueError("Either image_path or image_url must be provided")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {e}")

def tool_node(state: MessagesState) -> Dict:
    """Execute tools based on the agent's tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return {"messages": []}
    
    tools = {
        "vector_store": ChromaDBTool(data_path="./data"),
        "web_search": WebSearchTool(api_key="ed09d22d123a41cab542c563919882c1460269e2b19f9f8803a27c7af888f324")
    }
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.name
        args = json.loads(tool_call.arguments)
        
        if tool_name in tools:
            tool = tools[tool_name]
            if tool_name == "vector_store":
                result = tool.retrieve(**args)
            else:  # web_search
                result = tool.search(**args)
                
            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call.id
                )
            )
    
    return {"messages": results}

def call_model(state: MessagesState) -> Dict:
    """Call the language model to process the current state"""
    messages = state["messages"]
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    ).bind_tools([
        {
            "type": "function",
            "function": {
                "name": "vector_store",
                "description": "Retrieve documents from ChromaDB vector store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant documents"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using Google",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ])
    
    response = model.invoke(messages)
    return {"messages": [response]}

# Create and configure the workflow
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Example usage
    query = "What is machine learning?"
    
    # Initialize ChromaDB tool with chat history
    chroma_tool = ChromaDBTool(data_path="./data")
    
    # First try to find relevant information in vector store
    vector_results = chroma_tool.retrieve(query=query)
    
    # If no relevant results found, perform web search
    if not vector_results:
        web_tool = WebSearchTool(api_key="your_serpapi_key_here")
        search_results = web_tool.search(query=query)
        print("Web Search Results:", json.dumps(search_results, indent=2))
    else:
        print("Vector Store Results:", json.dumps(vector_results, indent=2))
    
    # Example of querying with chat history context
    response = chroma_tool.query("Tell me more about supervised learning")
    print("\nQuery Response:", response)
    
    # Display chat history
    print("\nChat History:")
    print(json.dumps(chroma_tool.chat_history.get_messages(), indent=2))