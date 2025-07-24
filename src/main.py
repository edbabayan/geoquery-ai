# Loaders
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.schema import Document

# Text Splitters
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

from langchain_experimental.text_splitter import SemanticChunker
# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# Vector Stores
from langchain.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import Chroma

# Retrievers
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Storage
from langchain.storage import InMemoryStore

# Prompts and Parsing
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Summarization
# Summarizer we'll use for Map Reduce

# Additional Libraries
import os  # noqa: F401
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
