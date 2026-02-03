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
            target_str = f"{match.group(1)} {match.group(2)}" # e.g. "table 11" (normalized)
            target_num = match.group(2)
            keyword_type = match.group(1) # table or figure
            
            # Scan ALL chunks for this specific reference
            # This is fast enough for <100 page documents
            for chunk in self.chunks:
                c_lower = chunk["content"].lower()
                # Check for "table 11" or "table.11"
                if f"{keyword_type} {target_num}" in c_lower or f"{keyword_type}.{target_num}" in c_lower:
                    explicit_matches.append(chunk)
                    
        # Add explicit matches first
        final_results = []
        
        if explicit_matches:
            final_results.extend(explicit_matches)
        
        # Then add retrieved semantic results (deduplicating)
        seen_content = set(c["content"] for c in final_results)
        
        # Standard semantic results
        for chunk in retrieved:
            if chunk["content"] not in seen_content:
                
                # Apply section boosting logic for semantic hits
                # Priority: Conclusion
                if "conclusion" in query_lower and chunk["section"] == "conclusion":
                    final_results.insert(len(explicit_matches), chunk) # Put right after exact matches
                
                # Priority: Results
                elif any(k in query_lower for k in ["result", "accuracy", "f1", "evaluation"]) and chunk["section"] in ["results", "evaluation"]:
                     final_results.append(chunk)
                else:
                     final_results.append(chunk) # End of list
                
                seen_content.add(chunk["content"])

        return final_results[:top_k]

