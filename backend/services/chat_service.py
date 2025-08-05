import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from google import genai
from google.genai import types

from backend.models import ChatMessage, ChatResponse
from backend.services.vector_service import VectorService
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chat_sessions: Dict[str, List[ChatMessage]] = {}
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    async def process_query(self, query: str, session_id: str) -> ChatResponse:
        """Process a chat query using RAG."""
        try:
            # Get previous conversation context
            conversation_context = self._get_conversation_context(session_id, query)
            
            # Determine query type for appropriate response strategy
            query_type = self._determine_query_type(query)
            
            # Adjust search parameters based on query type
            if query_type == "summary":
                top_k = 8  # Fewer, more diverse chunks for summary
            elif query_type == "detail":
                top_k = 20  # More chunks for comprehensive detail
            else:
                top_k = 15  # Default
            
            # Get relevant documents with very low threshold for name searches
            relevant_docs, scores = await self.vector_service.similarity_search(
                query=query,
                top_k=top_k,
                score_threshold=0.01
            )
            
            # If searching for specific names, always try text search fallback as well
            if self._is_name_query(query):
                logger.info(f"Detected name query: {query}")
                text_search_docs = await self.vector_service.text_search_fallback(query)
                if text_search_docs:
                    logger.info(f"Text search found {len(text_search_docs)} matches")
                    # Prioritize text search results by putting them first
                    relevant_docs = text_search_docs + relevant_docs[:10]  # Limit total results
                    scores = [0.9] * len(text_search_docs) + scores[:10]
                else:
                    logger.info("Text search found no matches")
            
            # Build context based on query type
            context = self._build_context(relevant_docs, query_type)
            has_context = len(relevant_docs) > 0
            
            # Check if query is asking about visual content
            visual_query = self._is_visual_query(query)
            
            # Generate response with conversation context and query type
            response_text = await self._generate_response(query, context, has_context, conversation_context, query_type, visual_query)
            
            # Extract sources
            sources = self._extract_sources(relevant_docs)
            
            # Calculate confidence
            confidence = self._calculate_confidence(scores, has_context)
            
            # Create response
            response = ChatResponse(
                response=response_text,
                sources=sources,
                confidence=confidence,
                has_context=has_context
            )
            
            # Update chat history
            self._update_chat_history(session_id, query, response_text, sources)
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return ChatResponse(
                response=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                has_context=False
            )
    
    def _build_context(self, relevant_docs: List[Dict], query_type: str = "default") -> str:
        """Build context string from relevant documents based on query type."""
        if not relevant_docs:
            return ""
        
        context_parts = []
        
        if query_type == "summary":
            # For summaries, use more chunks with longer content for comprehensive summaries
            max_chunks = 8
            for i, doc in enumerate(relevant_docs[:max_chunks]):
                source = doc["metadata"].get("source", "Unknown")
                content = doc["content"]
                # Use longer content for better summaries - increased from 300 to 800 characters
                summary_content = content[:800] + "..." if len(content) > 800 else content
                context_parts.append(f"[Source {i+1}: {source}]\n{summary_content}")
                
        elif query_type == "detail":
            # For details, use more chunks with full content
            max_chunks = 8
            for i, doc in enumerate(relevant_docs[:max_chunks]):
                source = doc["metadata"].get("source", "Unknown")
                content = doc["content"]
                score = doc["score"]
                context_parts.append(f"[Source {i+1}: {source} (relevance: {score:.2f})]\n{content}")
                
        else:
            # Default behavior
            max_chunks = 5
            for i, doc in enumerate(relevant_docs[:max_chunks]):
                source = doc["metadata"].get("source", "Unknown")
                content = doc["content"]
                score = doc["score"]
                context_parts.append(f"[Source {i+1}: {source} (relevance: {score:.2f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str, has_context: bool, conversation_context: str = "", query_type: str = "default", visual_query: bool = False) -> str:
        """Generate response using Gemini API."""
        try:
            if has_context:
                # Customize system prompt based on query type
                if query_type == "summary":
                    system_prompt = """You are a helpful project knowledge transfer assistant. The user is asking for a summary or overview.

Instructions:
1. Provide a comprehensive, well-structured summary based on the provided context
2. Include all key points and main topics with sufficient detail
3. Organize information logically with clear headings, subheadings, and bullet points
4. Highlight the most important information first, but include supporting details
5. Make the response thorough and complete - don't cut information short
6. Use proper formatting with sections and subsections for better readability
7. Include specific examples, data points, and technical details when available
8. Aim for completeness over brevity - users want full summaries, not abbreviated versions"""
                    
                elif query_type == "detail":
                    system_prompt = """You are a helpful project knowledge transfer assistant. The user is asking for detailed, comprehensive information.

Instructions:
1. Provide thorough, detailed information based on the provided context
2. Include specific data, numbers, and technical details when available
3. Quote relevant sections verbatim when helpful
4. Organize detailed information clearly with proper structure
5. Don't summarize - provide full information from the context
6. Include all relevant details that help answer the question completely"""
                    
                else:
                    base_instructions = """You are a helpful project knowledge transfer assistant. You help users understand project documentation and provide accurate information based on the provided context.

Instructions:
1. Answer the user's question based ONLY on the provided context
2. For name/role queries, carefully extract all mentions of people, their roles, and responsibilities
3. If asking about a specific person, include their title, department, and any actions they took
4. Be specific and cite relevant details from the context with exact quotes when helpful
5. If multiple sources provide different information, acknowledge this
6. Keep your response focused and helpful
7. For people searches, look for variations like first names, last names, titles, and roles
8. Do not make up information that's not in the context"""
                    
                    if visual_query:
                        system_prompt = base_instructions + """

VISUAL CONTENT HANDLING:
- Pay special attention to visual elements (images, diagrams, charts, figures)
- When describing visual content, include details from AI descriptions, captions, and OCR text
- Explain how visual elements relate to the surrounding text context
- If the user asks about visual content that isn't fully described, acknowledge the limitations
- Distinguish between different types of visual content (diagrams vs images vs charts)"""
                    else:
                        system_prompt = base_instructions

                # Build user prompt with conversation context if available
                prompt_parts = []
                
                if conversation_context:
                    prompt_parts.append(f"Previous conversation context:\n{conversation_context}\n")
                
                prompt_parts.append(f"Context from project documents:\n{context}\n")
                prompt_parts.append(f"User Question: {query}\n")
                prompt_parts.append("Please provide a helpful answer based on the context above. If this question references previous responses in our conversation, use that context to provide a more complete answer. If the context doesn't contain enough information to fully answer the question, please let the user know what information is missing.")
                
                user_prompt = "\n".join(prompt_parts)

            else:
                system_prompt = """You are a helpful project knowledge transfer assistant. The user has asked a question, but no relevant documents were found in the knowledge base."""

                user_prompt = f"""User Question: {query}

I don't have any relevant documents in my knowledge base to answer your question. Please consider:
1. Uploading relevant project documents that might contain this information
2. Rephrasing your question to be more specific
3. Checking if the information might be in documents you haven't uploaded yet

Is there anything else I can help you with regarding the documents you've already uploaded?"""

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user", 
                        parts=[types.Part(text=user_prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )
            
            return response.text if response.text else "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _extract_sources(self, relevant_docs: List[Dict]) -> List[str]:
        """Extract unique source filenames from relevant documents."""
        sources = set()
        for doc in relevant_docs:
            source = doc["metadata"].get("source", "Unknown")
            sources.add(source)
        return list(sources)
    
    def _calculate_confidence(self, scores: List[float], has_context: bool) -> float:
        """Calculate confidence score based on retrieval scores."""
        if not has_context or not scores:
            return 0.0
        
        # Average of top scores, weighted by position
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, score in enumerate(scores[:3]):  # Top 3 scores
            weight = 1.0 / (i + 1)  # Higher weight for better ranked results
            weighted_score += score * weight
            total_weight += weight
        
        confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _update_chat_history(self, session_id: str, query: str, response: str, sources: List[str]):
        """Update chat history for the session."""
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        # Add user message
        user_message = ChatMessage(
            role="user",
            content=query,
            timestamp=datetime.now(),
            sources=[]
        )
        
        # Add assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            sources=sources
        )
        
        self.chat_sessions[session_id].extend([user_message, assistant_message])
        
        # Keep only last 20 messages to prevent memory issues
        if len(self.chat_sessions[session_id]) > 20:
            self.chat_sessions[session_id] = self.chat_sessions[session_id][-20:]
    
    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session."""
        return self.chat_sessions.get(session_id, [])
    
    def clear_chat_history(self, session_id: str) -> None:
        """Clear chat history for a session."""
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            logger.info(f"Cleared chat history for session: {session_id}")
    
    def clear_all_sessions(self) -> None:
        """Clear all chat sessions."""
        self.chat_sessions.clear()
        logger.info("Cleared all chat sessions")
    
    def _is_name_query(self, query: str) -> bool:
        """Check if query is asking about a specific person or role."""
        name_indicators = ['who is', 'who are', 'what is', 'role of', 'position of', 
                          'lead', 'manager', 'developer', 'cto', 'qa', 'tester', 
                          'head of', 'responsible for', 'in charge of']
        query_lower = query.lower()
        
        # Check for name patterns (capitalized words that might be names)
        import re
        has_proper_names = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query))
        has_name_indicators = any(indicator in query_lower for indicator in name_indicators)
        
        # Check for specific names we know are in the documents
        known_names = ['ramesh', 'iyer', 'meera', 'nair', 'priya', 'deshmukh', 
                      'anjali', 'mukherjee', 'vishal', 'menon', 'devika', 'sharma',
                      'arjun', 'mehta', 'kavya', 'rathi', 'neeraj', 'kapoor',
                      'ayesha', 'khan']
        has_known_names = any(name in query_lower for name in known_names)
        
        return has_proper_names or has_name_indicators or has_known_names
    
    def _get_conversation_context(self, session_id: str, current_query: str) -> str:
        """Get relevant conversation context from previous messages."""
        if session_id not in self.chat_sessions or not self.chat_sessions[session_id]:
            return ""
        
        # Get recent messages (last 6 messages = 3 exchanges)
        recent_messages = self.chat_sessions[session_id][-6:]
        
        # Check if current query might reference previous responses
        if not self._query_references_previous_context(current_query):
            return ""
        
        # Build conversation context
        context_parts = []
        for msg in recent_messages:
            role = msg.role
            content = msg.content[:200]  # Limit length
            context_parts.append(f"{role.title()}: {content}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _query_references_previous_context(self, query: str) -> bool:
        """Check if query references previous conversation."""
        reference_indicators = [
            'who did you mention', 'what did you say', 'tell me more', 'more about',
            'what about', 'how about', 'and what', 'also', 'additionally',
            'that person', 'they', 'them', 'he', 'she', 'it', 'this',
            'previous', 'before', 'earlier', 'above', 'that', 'those'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in reference_indicators)
    
    def _is_visual_query(self, query: str) -> bool:
        """Check if query is asking about visual content."""
        visual_indicators = [
            'figure', 'diagram', 'image', 'picture', 'chart', 'graph', 
            'illustration', 'visual', 'show', 'display', 'screenshot',
            'drawing', 'sketch', 'flowchart', 'architecture', 'design',
            'what does', 'how does', 'show me', 'visualize', 'layout'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in visual_indicators)
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query to customize response strategy."""
        query_lower = query.lower()
        
        # Summary/overview keywords
        summary_keywords = [
            'summarize', 'summary', 'overview', 'brief', 'outline', 'synopsis',
            'main points', 'key points', 'highlights', 'quick overview',
            'give me a summary', 'what are the main', 'overall picture'
        ]
        
        # Detail/comprehensive keywords
        detail_keywords = [
            'details', 'detailed', 'full info', 'complete section', 'comprehensive',
            'in depth', 'thorough', 'complete details', 'full description',
            'tell me everything', 'all information', 'complete picture'
        ]
        
        # Check for summary intent
        if any(keyword in query_lower for keyword in summary_keywords):
            return "summary"
        
        # Check for detail intent
        if any(keyword in query_lower for keyword in detail_keywords):
            return "detail"
        
        return "default"
