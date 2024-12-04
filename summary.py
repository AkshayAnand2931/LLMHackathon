# Summarization Step
def summarize_documents(docs):
        """
        Summarizes the retrieved documents using an abstractive summarization model.

        Args:
            docs (str): The content of retrieved documents as a single string.

        Returns:
            str: A concise summary of the documents.
        """
        from transformers import pipeline

        # Load a pretrained summarization pipeline (e.g., BART or T5)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Process the documents for summarization (if too long, chunk into parts)
        if len(docs) > 1000:
            summaries = []
            # Chunking documents for summarization
            chunks = [docs[i:i+1000] for i in range(0, len(docs), 1000)]
            for chunk in chunks:
                summaries.append(summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"])
            return " ".join(summaries)
        else:
            summary = summarizer(docs, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
            return summary