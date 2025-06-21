import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

# ChromaDB integration
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

# Load environment variables
load_dotenv()

class SMSRAGPipeline:
    def __init__(self, data_dir: str = "2025-05", db_path: str = "chroma_db"):
        self.data_dir = Path(data_dir)
        self.document_store = ChromaDocumentStore(persist_path=db_path)
        
        # Verify OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    def load_and_process_conversations(self) -> List[Document]:
        """Load JSON files and group messages by conversation ID"""
        print("Loading SMS conversations...")
        
        conversations = {}
        
        # Process all JSON files in the data directory
        for json_file in self.data_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract conversation and message data
                conv_data = data.get("conversation", {})
                conv_id = conv_data.get("id")
                message = conv_data.get("message", {})
                
                if not conv_id or not message:
                    continue
                
                # Initialize conversation if not exists
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "messages": [],
                        "participants": set()
                    }
                
                # Extract message details
                timestamp = message.get("timestamp", 0)
                from_field = message.get("from_field", {})
                to_fields = message.get("to_fields", [])
                body = message.get("body", "").replace("<br>", "\n").replace("&#39;", "'")
                
                # Track participants
                from_number = from_field.get("name") or from_field.get("id", "Unknown")
                if from_number:
                    conversations[conv_id]["participants"].add(from_number)
                
                for to_field in to_fields:
                    to_number = to_field.get("name") or to_field.get("id", "Unknown")
                    if to_number:
                        conversations[conv_id]["participants"].add(to_number)
                
                # Add message to conversation
                to_numbers = []
                for field in to_fields:
                    to_num = field.get("name") or field.get("id", "Unknown")
                    if to_num:
                        to_numbers.append(to_num)
                
                conversations[conv_id]["messages"].append({
                    "timestamp": timestamp,
                    "from": from_number or "Unknown",
                    "to": to_numbers,
                    "body": body,
                    "message_id": message.get("id", "")
                })
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        print(f"Found {len(conversations)} conversations")
        return self._create_documents(conversations)
    
    def _create_documents(self, conversations: Dict) -> List[Document]:
        """Convert conversations to Haystack Documents"""
        documents = []
        
        for conv_id, conv_data in conversations.items():
            # Sort messages by timestamp
            messages = sorted(conv_data["messages"], key=lambda x: x["timestamp"])
            participants = [p for p in conv_data["participants"] if p and p != "Unknown"]
            
            # Build conversation text
            conversation_text = []
            for msg in messages:
                timestamp_str = datetime.fromtimestamp(msg["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                to_list = ", ".join(msg["to"]) if msg["to"] else "Unknown"
                from_name = msg["from"] if msg["from"] else "Unknown"
                conversation_text.append(f"[{timestamp_str}] {from_name} → {to_list}: {msg['body']}")
            
            # Create document content
            participants_str = ", ".join(participants) if participants else "Unknown participants"
            content = f"Conversation between: {participants_str}\n\n" + "\n".join(conversation_text)
            
            # Create metadata (ChromaDB requires string values for lists)
            metadata = {
                "conversation_id": conv_id,
                "participants": ", ".join(participants) if participants else "Unknown",
                "message_count": len(messages),
                "start_time": min(msg["timestamp"] for msg in messages) if messages else 0,
                "end_time": max(msg["timestamp"] for msg in messages) if messages else 0
            }
            
            documents.append(Document(content=content, meta=metadata))
        
        print(f"Created {len(documents)} documents")
        return documents
    
    def build_indexing_pipeline(self) -> Pipeline:
        """Build pipeline for indexing documents"""
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("splitter", DocumentSplitter(
            split_by="word",
            split_length=500,
            split_overlap=50,
            split_threshold=0
        ))
        indexing_pipeline.add_component("embedder", OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        ))
        indexing_pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")
        
        return indexing_pipeline
    
    def build_query_pipeline(self) -> Pipeline:
        """Build RAG pipeline for querying"""
        # Create prompt template following Haystack documentation pattern
        prompt_template = [
            ChatMessage.from_system("You are a helpful assistant analyzing SMS conversations. Answer questions based on the provided conversation data."),
            ChatMessage.from_user(
                """Given these SMS conversations, answer the question.
                
                Conversations:
                {% for doc in documents %}
                {{ doc.content }}
                ---
                {% endfor %}
                
                Question: {{question}}
                Answer:"""
            )
        ]
        
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        ))
        query_pipeline.add_component("retriever", ChromaEmbeddingRetriever(
            self.document_store, 
            top_k=5
        ))
        query_pipeline.add_component("prompt_builder", ChatPromptBuilder(
            template=prompt_template,
            required_variables={"question", "documents"}
        ))
        query_pipeline.add_component("llm", OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="gpt-4-turbo"
        ))
        
        # Connect components
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        query_pipeline.connect("retriever", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "llm.messages")
        
        return query_pipeline
    
    def clear_database(self):
        """Clear all documents from the database"""
        try:
            # Delete all documents
            all_docs = self.document_store.filter_documents()
            if all_docs:
                doc_ids = [doc.id for doc in all_docs]
                self.document_store.delete_documents(doc_ids)
                print(f"Cleared {len(doc_ids)} documents from database")
        except Exception as e:
            print(f"Error clearing database: {e}")
    
    def index_documents(self, force_reindex=False):
        """Load and index all SMS conversations"""
        # Check if already indexed
        if not force_reindex and self.document_store.count_documents() > 0:
            print(f"Found {self.document_store.count_documents()} existing documents in ChromaDB")
            print("Use force_reindex=True to rebuild the index")
            return
        
        # Load conversations and create documents
        documents = self.load_and_process_conversations()
        
        if not documents:
            print("No documents to index!")
            return
        
        # Build and run indexing pipeline
        indexing_pipeline = self.build_indexing_pipeline()
        print("Indexing documents...")
        result = indexing_pipeline.run({"splitter": {"documents": documents}})
        
        # Handle result properly - documents_written might be int or list
        documents_written = result.get("writer", {}).get("documents_written", 0)
        if isinstance(documents_written, int):
            indexed_count = documents_written
        else:
            indexed_count = len(documents_written) if documents_written else 0
        
        print(f"Successfully indexed {indexed_count} document chunks from {len(documents)} conversations!")
    
    def ask_question(self, question: str) -> str:
        """Query the RAG pipeline with a question"""
        query_pipeline = self.build_query_pipeline()
        
        result = query_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        
        return result["llm"]["replies"][0].text
    
    def interactive_chat(self):
        """Run interactive command-line interface"""
        print("\n🤖 SMS RAG Assistant Ready!")
        print("Ask questions about your SMS conversations. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("🔍 Searching conversations...")
                answer = self.ask_question(question)
                print(f"\n💬 Answer: {answer}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    try:
        # Initialize RAG pipeline
        rag = SMSRAGPipeline()
        
        # Index documents
        rag.index_documents()
        
        # Start interactive chat
        rag.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")

if __name__ == "__main__":
    main() 