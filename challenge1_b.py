#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Any, Tuple
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaDrivenAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the analyzer with a lightweight sentence transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF, organized by page number."""
        page_texts = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        page_texts[page_num] = text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        return page_texts
    
    def identify_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Identify sections within a page using heuristics."""
        sections = []
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 50:
                continue
            
            # Heuristics to identify section titles
            lines = paragraph.split('\n')
            potential_title = lines[0].strip()
            
            # Check if first line looks like a heading
            is_heading = (
                len(potential_title) < 100 and
                (potential_title.isupper() or 
                 potential_title.istitle() or
                 re.match(r'^[\d\.]+\s+[A-Z]', potential_title) or
                 re.match(r'^[A-Z][^.!?]*$', potential_title))
            )
            
            if is_heading and len(lines) > 1:
                title = potential_title
                content = '\n'.join(lines[1:])
            else:
                # Use first sentence or first 50 chars as title
                sentences = sent_tokenize(paragraph)
                if sentences:
                    title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
                else:
                    title = paragraph[:50] + "..."
                content = paragraph
            
            sections.append({
                'title': title,
                'content': content,
                'page': page_num,
                'section_id': f"page_{page_num}_section_{i}"
            })
        
        return sections
    
    def create_persona_query(self, persona: str, job_to_be_done: str) -> str:
        """Create a search query based on persona and job."""
        return f"{persona}: {job_to_be_done}"
    
    def calculate_relevance_scores(self, sections: List[Dict], query: str) -> List[Tuple[Dict, float]]:
        """Calculate relevance scores for sections based on the query."""
        if not sections:
            return []
        
        # Prepare texts for embedding
        section_texts = []
        for section in sections:
            # Combine title and content for better context
            combined_text = f"{section['title']} {section['content']}"
            section_texts.append(combined_text)
        
        try:
            # Get embeddings
            query_embedding = self.model.encode([query])
            section_embeddings = self.model.encode(section_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, section_embeddings)[0]
            
            # Create scored sections
            scored_sections = []
            for i, section in enumerate(sections):
                scored_sections.append((section, float(similarities[i])))
            
            return scored_sections
        except Exception as e:
            logger.error(f"Error calculating relevance scores: {e}")
            return [(section, 0.0) for section in sections]
    
    def extract_subsections(self, section_content: str, max_subsections: int = 3) -> List[Dict[str, str]]:
        """Extract relevant subsections from a section."""
        sentences = sent_tokenize(section_content)
        
        if len(sentences) <= max_subsections:
            return [{'text': sent, 'refined_text': sent} for sent in sentences]
        
        # Simple extractive approach - select diverse sentences
        subsections = []
        
        # Take first sentence (often contains key info)
        if sentences:
            subsections.append({
                'text': sentences[0],
                'refined_text': sentences[0]
            })
        
        # Take middle sentences based on length and keyword density
        if len(sentences) > 2:
            middle_idx = len(sentences) // 2
            subsections.append({
                'text': sentences[middle_idx],
                'refined_text': sentences[middle_idx]
            })
        
        # Take last sentence if it's substantial
        if len(sentences) > 1 and len(sentences[-1]) > 50:
            subsections.append({
                'text': sentences[-1],
                'refined_text': sentences[-1]
            })
        
        return subsections[:max_subsections]
    
    def process_document_collection(self, input_dir: str, output_dir: str):
        """Process all documents in the input directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        input_json = input_path / "input.json"
        if not input_json.exists():
         logger.error("input.json file is missing in input directory")
         return

        with open(input_json, "r") as f:
         input_data = json.load(f)

        persona = input_data["persona"]["role"]
        job_to_be_done = input_data["job_to_be_done"]["task"]

        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.error("No PDF files found in input directory")
            return
        
        logger.info(f"Processing {len(pdf_files)} documents for persona: {persona}")
        
        # Extract all sections from all documents
        all_sections = []
        document_info = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}")
            
            page_texts = self.extract_text_from_pdf(str(pdf_file))
            document_info.append({
                'filename': pdf_file.name,
                'pages': len(page_texts)
            })
            
            for page_num, text in page_texts.items():
                sections = self.identify_sections(text, page_num)
                for section in sections:
                    section['document'] = pdf_file.name
                    all_sections.append(section)
        
        # Create query and calculate relevance
        query = self.create_persona_query(persona, job_to_be_done)
        scored_sections = self.calculate_relevance_scores(all_sections, query)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sections (limit to prevent overwhelming output)
        top_sections = scored_sections[:10]  # Top 10 most relevant sections
        
        # Prepare output
        extracted_sections = []
        subsection_analysis = []
        
        for rank, (section, score) in enumerate(top_sections, 1):
            # Add to extracted sections
            extracted_sections.append({
                'document': section['document'],
                'page_number': section['page'],
                'section_title': section['title'],
                'importance_rank': rank
            })
            
            # Extract subsections
            subsections = self.extract_subsections(section['content'])
            
            for subsection in subsections:
                subsection_analysis.append({
                    'document': section['document'],
                    'page_number': section['page'],
                    'refined_text': subsection['refined_text']
                })
        
        # Create final output
        output_data = {
            'metadata': {
                'input_documents': [doc['filename'] for doc in document_info],
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }
        
        # Write output
        output_file = output_path / "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to {output_file}")

def main():
    """Main execution function."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    try:
        analyzer = PersonaDrivenAnalyzer()
        analyzer.process_document_collection(input_dir, output_dir)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()