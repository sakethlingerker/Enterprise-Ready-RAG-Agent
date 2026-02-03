import re

def recursive_split(text, chunk_size, overlap):
    """
    Recursively splits text by separators to find the best split point.
    Separators: Paragraphs (\n\n), Sentences (. ), Words ( ).
    """
    if len(text) <= chunk_size:
        return [text]
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    for sep in separators:
        if sep == "":
            # Fallback to character splitting
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
        
        splits = text.split(sep)
        if len(splits) > 1:
            chunks = []
            current_chunk = ""
            
            for split in splits:
                if len(current_chunk) + len(split) + len(sep) <= chunk_size:
                    current_chunk += (sep + split) if current_chunk else split
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = split
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Verify if any chunk is still too large, if so, recurse
            final_chunks = []
            for c in chunks:
                if len(c) > chunk_size:
                    final_chunks.extend(recursive_split(c, chunk_size, overlap))
                else:
                    final_chunks.append(c)
            
            return final_chunks
            
    return [text]

def chunk_text(sections, chunk_size=800, overlap=100):
    chunks = []

    for sec in sections:
        text = sec["content"]
        section_name = sec.get("section", "unknown")
        
        # Use recursive splitting instead of fixed stride
        text_chunks = recursive_split(text, chunk_size, overlap)
        
        for content in text_chunks:
            chunks.append({
                "content": content,
                "page": sec["page"],
                "section": section_name
            })
            
    return chunks
