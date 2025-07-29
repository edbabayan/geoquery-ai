import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import CrossEncoder
import torch


class EnhancedTableRetriever:
    """Enhanced retrieval system with semantic hybrid search, cross-encoder reranking, and query expansion"""
    
    def __init__(self, 
                 documents: List[Document],
                 embed_model: str = "text-embedding-3-large",
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the enhanced retriever with documents and models
        
        Args:
            documents: List of Document objects with table descriptions
            embed_model: OpenAI embedding model name
            cross_encoder_model: Hugging Face cross-encoder model for reranking
        """
        self.documents = documents
        self.embed_model = OpenAIEmbeddings(model=embed_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.query_expansion_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        
        # Initialize retrievers
        self._setup_retrievers()
        
    def _setup_retrievers(self):
        """Set up BM25 and vector retrievers"""
        # BM25 for keyword matching
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        
        # Vector store for semantic search
        self.vectorstore = Chroma.from_documents(
            self.documents, 
            self.embed_model, 
            collection_name="enhanced_table_search"
        )
        self.vector_retriever = self.vectorstore.as_retriever()
        
    def _calculate_hybrid_scores(self, 
                                query: str, 
                                bm25_results: List[Document], 
                                vector_results: List[Document],
                                alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Calculate hybrid scores combining BM25 and vector similarity
        
        Args:
            query: User query
            bm25_results: Results from BM25 retriever
            vector_results: Results from vector retriever
            alpha: Weight for BM25 vs vector scores (0-1)
        
        Returns:
            List of (document, score) tuples
        """
        doc_scores = {}
        
        # Normalize and combine BM25 scores
        for i, doc in enumerate(bm25_results):
            table_name = doc.metadata.get('table_name', '')
            # Higher rank = higher score (using reciprocal rank)
            bm25_score = 1.0 / (i + 1)
            doc_scores[table_name] = {
                'doc': doc,
                'bm25': bm25_score,
                'vector': 0
            }
        
        # Normalize and combine vector scores
        for i, doc in enumerate(vector_results):
            table_name = doc.metadata.get('table_name', '')
            vector_score = 1.0 / (i + 1)
            
            if table_name in doc_scores:
                doc_scores[table_name]['vector'] = vector_score
            else:
                doc_scores[table_name] = {
                    'doc': doc,
                    'bm25': 0,
                    'vector': vector_score
                }
        
        # Calculate final scores
        final_scores = []
        for table_name, scores in doc_scores.items():
            # Hybrid score
            hybrid_score = alpha * scores['bm25'] + (1 - alpha) * scores['vector']
            final_scores.append((scores['doc'], hybrid_score))
        
        # Sort by score descending
        return sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original user query
            
        Returns:
            List of expanded queries
        """
        prompt = PromptTemplate(
            template="""You are an oil and gas data expert. Given a user query about oil and gas data, 
            generate 2-3 alternative phrasings or expansions that capture the same intent but use different keywords.
            Focus on industry-specific synonyms and related terms.
            
            For example:
            - "production" could be expanded to include "output", "yield", "flow rate"
            - "well" could include "wellbore", "borehole"
            - "pressure" could include "PSI", "bar", "atmospheric pressure"
            
            Original query: {query}
            
            Output as JSON array of strings (just the expanded queries, not the original):""",
            input_variables=["query"]
        )
        
        chain = prompt | self.query_expansion_llm | JsonOutputParser()
        
        try:
            expansions = chain.invoke({"query": query})
            return [query] + expansions  # Include original query
        except:
            return [query]  # Fallback to original query only
    
    def rerank_with_cross_encoder(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents using cross-encoder for better relevance
        
        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if len(documents) <= top_k:
            return documents
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for doc in documents:
            # Combine table name and description for better context
            table_name = doc.metadata.get('table_name', '')
            doc_text = f"Table: {table_name}. {doc.page_content}"
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by cross-encoder scores
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, score in doc_scores[:top_k]]
    
    def retrieve(self, query: str, top_k: int = 5, use_query_expansion: bool = True, 
                 use_reranking: bool = True, alpha: float = 0.5) -> List[Document]:
        """
        Main retrieval method with all enhancements
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_query_expansion: Whether to expand the query
            use_reranking: Whether to apply cross-encoder reranking
            alpha: Weight for BM25 vs vector scores (0-1)
            
        Returns:
            List of retrieved documents
        """
        # Expand query if enabled
        if use_query_expansion:
            expanded_queries = self.expand_query(query)
        else:
            expanded_queries = [query]
        
        all_bm25_results = []
        all_vector_results = []
        
        # Retrieve for each expanded query
        for exp_query in expanded_queries:
            # Get more results for reranking
            retrieval_size = top_k * 3 if use_reranking else top_k
            
            self.bm25_retriever.k = retrieval_size
            bm25_results = self.bm25_retriever.invoke(exp_query)
            all_bm25_results.extend(bm25_results)
            
            self.vector_retriever.search_kwargs = {"k": retrieval_size}
            vector_results = self.vector_retriever.invoke(exp_query)
            all_vector_results.extend(vector_results)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_bm25 = []
        for doc in all_bm25_results:
            table_name = doc.metadata.get('table_name', '')
            if table_name not in seen:
                seen.add(table_name)
                unique_bm25.append(doc)
        
        seen = set()
        unique_vector = []
        for doc in all_vector_results:
            table_name = doc.metadata.get('table_name', '')
            if table_name not in seen:
                seen.add(table_name)
                unique_vector.append(doc)
        
        # Calculate hybrid scores
        scored_docs = self._calculate_hybrid_scores(query, unique_bm25, unique_vector, alpha)
        
        # Get top documents before reranking
        candidates = [doc for doc, score in scored_docs[:top_k * 2 if use_reranking else top_k]]
        
        # Rerank with cross-encoder if enabled
        if use_reranking and len(candidates) > 0:
            final_docs = self.rerank_with_cross_encoder(query, candidates, top_k)
        else:
            final_docs = candidates[:top_k]
        
        return final_docs
    
    def batch_retrieve(self, queries: List[str], top_k: int = 5, **kwargs) -> Dict[str, List[Document]]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: List of queries
            top_k: Number of documents to retrieve per query
            **kwargs: Additional arguments for retrieve method
            
        Returns:
            Dictionary mapping queries to retrieved documents
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k, **kwargs)
        return results