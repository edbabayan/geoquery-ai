"""
Default retriever enhanced with cross-encoder reranking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from retrieval_models import retriever_default, ground_truth_renamed
from dotenv import load_dotenv
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize cross-encoder model for reranking
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)

# Create a proper LangChain retriever class
class DefaultRetrieverK10(BaseRetriever):
    """Wrapper to make default retriever return 10 documents for reranking"""
    
    base_retriever: object
    
    def __init__(self, base_retriever, **kwargs):
        super().__init__(base_retriever=base_retriever, **kwargs)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents with k=10 for reranking"""
        # Get 10 documents for reranking (more than the final 5)
        docs = self.base_retriever.invoke(query)
        # Return up to 10 docs, or all if less than 10
        return docs[:10] if len(docs) >= 10 else docs

# Create base retriever with k=10 for reranking
default_retriever_k10 = DefaultRetrieverK10(retriever_default)

# Create compression retriever with cross-encoder reranking
retriever_default_with_reranker = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=default_retriever_k10
)

# Export for evaluation
__all__ = ['retriever_default_with_reranker', 'ground_truth_renamed']