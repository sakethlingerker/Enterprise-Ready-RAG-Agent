from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [c["content"] for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query, top_k=6):
        """
        Retrieves relevant chunks using semantic search + keyword boosting.
        """
        query_embedding = self.model.encode([query]).astype("float32")
        # Search deeper to find candidates
        _, indices = self.index.search(query_embedding, top_k * 3)

        retrieved = [self.chunks[i] for i in indices[0]]
        query_lower = query.lower()
        
        # 🎯 Keyword-based Prioritization (Regex Force)
        prioritized_chunks = []
        
        # Regex to find "Table 11", "Figure 3", "Tab. 5" etc.
        import re
        # Match "table 11", "table.11", "tab 11"
        match = re.search(r"(table|tab|figure|fig)\.?\s*(\d+)", query_lower)
        
        explicit_matches = []
        if match:
            # User specifically asked for Table X or Figure X
            target_num = match.group(2)
            keyword_type = match.group(1) # table or figure
            
            # Scan ALL chunks for this specific reference
            for chunk in self.chunks:
                # Check both child content and larger parent content if available
                c_lower = chunk.get("parent_content", chunk["content"]).lower()
                # Check for "table 11" or "table.11"
                if f"{keyword_type} {target_num}" in c_lower or f"{keyword_type}.{target_num}" in c_lower:
                    explicit_matches.append(chunk)
                    
        # Add explicit matches first
        final_results = []
        
        if explicit_matches:
            final_results.extend(explicit_matches)
        
        # Track seen parent IDs to prevent duplicates
        seen_parent_ids = set()
        for chunk in final_results:
            p_id = chunk.get("parent_id")
            if p_id:
                seen_parent_ids.add(p_id)
        
        # Deduplication tracker for parent content strings
        seen_content = set(c.get("parent_content", c["content"]) for c in final_results)
        
        # Standard semantic results
        for chunk in retrieved:
            p_id = chunk.get("parent_id")
            chunk_content = chunk.get("parent_content", chunk["content"])
            
            # Deduplicate by parent ID
            if p_id and p_id in seen_parent_ids:
                continue
                
            if chunk_content not in seen_content:
                # Apply section boosting logic for semantic hits
                # Priority: Conclusion
                if "conclusion" in query_lower and chunk["section"] == "conclusion":
                    final_results.insert(len(explicit_matches), chunk) # Put right after exact matches
                
                # Priority: Results
                elif any(k in query_lower for k in ["result", "accuracy", "f1", "evaluation"]) and chunk["section"] in ["results", "evaluation"]:
                     final_results.append(chunk)
                else:
                     final_results.append(chunk) # End of list
                
                seen_content.add(chunk_content)
                if p_id:
                    seen_parent_ids.add(p_id)

        return final_results[:top_k]

