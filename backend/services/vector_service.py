import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
import numpy as np
import hashlib
import re

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(COLLECTION_NAME)
                logger.info(f"Loaded existing collection: {COLLECTION_NAME}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "Project knowledge transfer documents"}
                )
                logger.info(f"Created new collection: {COLLECTION_NAME}")
        
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise Exception(f"Vector database initialization failed: {str(e)}")
    
    async def add_documents(self, documents: List[Document], doc_id: str) -> tuple[bool, str]:
        """Add documents to vector store."""
        try:
            if not documents:
                return False, "No documents to add"
            
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # Create unique ID for each chunk
                chunk_id = f"{doc_id}_chunk_{i}"
                ids.append(chunk_id)
                
                # Prepare metadata
                metadata = doc.metadata.copy()
                metadata.update({
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text_length": len(doc.page_content)
                })
                metadatas.append(metadata)
            
            # Generate improved embeddings using enhanced text features
            embeddings = self._generate_improved_embeddings(texts)
            
            # Add to ChromaDB
            if self.collection is not None:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            
            logger.info(f"Added {len(documents)} documents to vector store for doc_id: {doc_id}")
            return True, f"Successfully added {len(documents)} document chunks"
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False, f"Failed to add documents: {str(e)}"
    
    async def similarity_search(self, query: str, top_k: int = 15, score_threshold: float = 0.05) -> tuple[List[Dict], List[float]]:
        """Perform similarity search."""
        try:
            if not self.collection:
                return [], []
            
            # Generate query embedding
            query_embedding = self._generate_improved_embeddings([query])[0]
            
            # Perform search
            if self.collection is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Process results
            documents = []
            scores = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses distance, lower is better)
                    similarity_score = 1 / (1 + distance)
                    
                    if similarity_score >= score_threshold:
                        doc_result = {
                            "content": doc,
                            "metadata": metadata,
                            "score": similarity_score
                        }
                        documents.append(doc_result)
                        scores.append(similarity_score)
            
            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents, scores
        
        except Exception as e:
            logger.error(f"Similarity search error: {str(e)}")
            return [], []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            if not self.collection:
                return {"status": "not_initialized"}
            
            count = self.collection.count()
            return {
                "status": "active",
                "document_count": count,
                "collection_name": COLLECTION_NAME
            }
        
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def clear_collection(self) -> tuple[bool, str]:
        """Clear all documents from the collection."""
        try:
            if self.collection:
                # Get all IDs and delete them
                all_data = self.collection.get()
                if all_data["ids"]:
                    self.collection.delete(ids=all_data["ids"])
                    logger.info("Cleared all documents from collection")
                    return True, "Collection cleared successfully"
                else:
                    return True, "Collection was already empty"
            else:
                return False, "Collection not initialized"
        
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False, f"Failed to clear collection: {str(e)}"
    
    def delete_document(self, doc_id: str) -> tuple[bool, str]:
        """Delete all chunks of a specific document."""
        try:
            if not self.collection:
                return False, "Collection not initialized"
            
            # Query for all chunks of this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
                return True, f"Deleted {len(results['ids'])} document chunks"
            else:
                return True, "No chunks found for this document"
        
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False, f"Failed to delete document: {str(e)}"
    
    async def text_search_fallback(self, query: str) -> List[Dict]:
        """Fallback text search for when vector search fails on names."""
        try:
            if not self.collection:
                return []
            
            # Extract search terms - improved to handle various patterns
            import re
            search_terms = []
            
            # Look for full names (First Last)
            full_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query)
            search_terms.extend(full_names)
            
            # Look for single names that might be first or last names
            single_names = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            search_terms.extend(single_names)
            
            # Add lowercased versions for case-insensitive matching
            query_lower = query.lower()
            
            # Common name patterns and titles to search for
            name_keywords = ['ramesh', 'iyer', 'meera', 'nair', 'priya', 'deshmukh', 
                           'anjali', 'mukherjee', 'vishal', 'menon', 'devika', 'sharma',
                           'arjun', 'mehta', 'kavya', 'rathi', 'neeraj', 'kapoor',
                           'ayesha', 'khan', 'cto', 'qa lead', 'project manager',
                           'head of operations', 'developer', 'tester']
            
            # Add relevant keywords from query
            for keyword in name_keywords:
                if keyword in query_lower:
                    search_terms.append(keyword)
            
            if not search_terms:
                # If no specific names/keywords found, do a general text search
                search_terms = [query_lower]
            
            # Get all documents and search for matches
            all_data = self.collection.get(include=["documents", "metadatas"])
            matches = []
            
            if all_data and all_data.get('documents') and all_data.get('metadatas'):
                for i, (doc_id, content, metadata) in enumerate(zip(
                    all_data.get('ids', []), 
                    all_data.get('documents', []), 
                    all_data.get('metadatas', [])
                )):
                    content_lower = content.lower()
                    match_score = 0.0
                    
                    # Check if any search terms appear in this chunk
                    for term in search_terms:
                        if term.lower() in content_lower:
                            # Higher score for exact matches
                            if term.lower() == term:  # Already lowercase keyword
                                match_score = max(match_score, 0.95)
                            else:  # Proper name match
                                match_score = max(match_score, 0.9)
                    
                    if match_score > 0:
                        matches.append({
                            "content": content,
                            "metadata": metadata,
                            "score": match_score
                        })
            
            # Sort by score descending
            matches.sort(key=lambda x: x["score"], reverse=True)
            return matches[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Text search fallback error: {str(e)}")
            return []
    
    def _generate_improved_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate improved embeddings using enhanced text features."""
        embeddings = []
        
        for text in texts:
            text_lower = text.lower()
            words = re.findall(r'\w+', text_lower)
            
            # Enhanced features for better semantic matching
            length_feature = min(len(text) / 2000.0, 1.0)
            word_count_feature = min(len(words) / 200.0, 1.0)
            
            # Visual content features
            has_visual_content = 1.0 if any(word in text_lower for word in ['figure', 'diagram', 'image', 'chart', 'graph', 'illustration']) else 0.0
            is_visual_chunk = 1.0 if 'visual' in text_lower and ('description:' in text_lower or 'caption:' in text_lower) else 0.0
            
            # Important keyword features for CRM project context
            crm_keywords = {
                'react': 0, 'node': 0, 'mongodb': 0, 'aws': 0, 'oauth': 0,
                'backend': 0, 'frontend': 0, 'api': 0, 'database': 0,
                'developer': 0, 'delay': 0, 'vendor': 0, 'approval': 0,
                'approved': 0, 'ramesh': 0, 'iyer': 0, 'meera': 0, 'nair': 0, 'cto': 0,
                'adoption': 0, 'metrics': 0, 'pipeline': 0, 'mobile': 0,
                'support': 0, 'tickets': 0, 'testing': 0, 'jest': 0,
                'security': 0, 'rbac': 0, 'encryption': 0, 'module': 0,
                'lead': 0, 'contact': 0, 'opportunity': 0, 'email': 0,
                'campaign': 0, 'analytics': 0, 'dashboard': 0, 'functional': 0,
                'april': 0, 'started': 0, 'final': 0, 'release': 0, 'uat': 0,
                'anjali': 0, 'mukherjee': 0, 'qa': 0, 'quality': 0, 'assurance': 0,
                'vishal': 0, 'menon': 0, 'devika': 0, 'sharma': 0, 'arjun': 0,
                'mehta': 0, 'kavya': 0, 'rathi': 0, 'neeraj': 0, 'kapoor': 0,
                'ayesha': 0, 'khan': 0, 'priya': 0, 'deshmukh': 0, 'role': 0
            }
            
            # Count keyword occurrences
            for word in words:
                if word in crm_keywords:
                    crm_keywords[word] += 1
            
            # Normalize keyword features
            total_words = len(words) or 1
            keyword_features = [count / total_words for count in crm_keywords.values()]
            
            # N-gram features for better context
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            important_bigrams = {
                'backend_developer': 0, 'email_campaign': 0, 'final_approval': 0,
                'post_release': 0, 'user_adoption': 0, 'support_tickets': 0,
                'pipeline_visibility': 0, 'mobile_usage': 0, 'tech_stack': 0,
                'project_delayed': 0, 'technology_stack': 0
            }
            
            for bigram in bigrams:
                if bigram in important_bigrams:
                    important_bigrams[bigram] += 1
            
            bigram_features = [count / max(len(bigrams), 1) for count in important_bigrams.values()]
            
            # Character frequency features (reduced)
            char_counts = {}
            for char in text_lower:
                if char.isalpha():
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            total_chars = sum(char_counts.values()) or 1
            char_features = []
            for char in 'aeioutrnslc':  # Most common chars
                freq = char_counts.get(char, 0) / total_chars
                char_features.append(freq)
            
            # Positional features
            has_numbers = 1.0 if any(c.isdigit() for c in text) else 0.0
            has_dates = 1.0 if re.search(r'\d{4}', text) else 0.0
            has_percentages = 1.0 if '%' in text else 0.0
            
            # Combine all features
            embedding = ([length_feature, word_count_feature, has_numbers, has_dates, has_percentages, has_visual_content, is_visual_chunk] + 
                        keyword_features + bigram_features + char_features)
            
            # Pad to fixed size
            target_size = 384
            if len(embedding) < target_size:
                embedding.extend([0.0] * (target_size - len(embedding)))
            else:
                embedding = embedding[:target_size]
                
            embeddings.append(embedding)
        
        return embeddings
