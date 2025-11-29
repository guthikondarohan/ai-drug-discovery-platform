"""
PubMed API Client for Real-Time Literature Search

Integrates with NCBI E-utilities to fetch scientific papers
related to molecules and compounds.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time
from datetime import datetime


class PubMedAPI:
    """
    Client for PubMed/NCBI E-utilities API.
    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str = "research@example.com", rate_limit: float = 0.34):
        """
        Args:
            email: Your email (NCBI requirement)
            rate_limit: Seconds between requests (max 3/sec without API key)
        """
        self.email = email
        self.rate_limit = rate_limit
        self.last_request = 0
    
    def _wait_for_rate_limit(self):
        """Ensure compliance with NCBI rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def search(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search term (e.g., "aspirin", "ibuprofen mechanism")
            max_results: Maximum papers to return
        
        Returns:
            List of PubMed IDs (PMIDs)
        """
        self._wait_for_rate_limit()
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = requests.get(
                f"{self.BASE_URL}esearch.fcgi",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                return data['esearchresult']['idlist']
            return []
        
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for papers.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of paper details (title, authors, abstract, etc.)
        """
        if not pmids:
            return []
        
        self._wait_for_rate_limit()
        
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email
        }
        
        try:
            response = requests.get(
                f"{self.BASE_URL}efetch.fcgi",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            
            return self._parse_pubmed_xml(response.text)
        
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall('.//PubmedArticle'):
                paper = {}
                
                # PMID
                pmid_elem = article.find('.//PMID')
                paper['pmid'] = pmid_elem.text if pmid_elem is not None else 'N/A'
                
                # Title
                title_elem = article.find('.//ArticleTitle')
                paper['title'] = title_elem.text if title_elem is not None else 'No title'
                
                # Authors
                authors = []
                for author in article.findall('.//Author'):
                    lastname = author.find('LastName')
                    forename = author.find('ForeName')
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)
                paper['authors'] = authors
                
                # Journal
                journal_elem = article.find('.//Journal/Title')
                paper['journal'] = journal_elem.text if journal_elem is not None else 'Unknown'
                
                # Year
                year_elem = article.find('.//PubDate/Year')
                paper['year'] = year_elem.text if year_elem is not None else 'N/A'
                
                # Abstract
                abstract_parts = article.findall('.//AbstractText')
                if abstract_parts:
                    abstract = ' '.join([
                        part.text for part in abstract_parts 
                        if part.text is not None
                    ])
                    paper['abstract'] = abstract
                else:
                    paper['abstract'] = 'No abstract available.'
                
                # DOI
                doi_elem = article.find('.//ArticleId[@IdType="doi"]')
                paper['doi'] = doi_elem.text if doi_elem is not None else None
                
                # PubMed URL
                paper['url'] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"
                
                papers.append(paper)
        
        except Exception as e:
            print(f"XML parsing error: {e}")
        
        return papers
    
    def search_by_compound(self, compound_name: str, max_results: int = 10) -> List[Dict]:
        """
        Search papers related to a specific compound.
        
        Args:
            compound_name: Name of compound (e.g., "aspirin", "ibuprofen")
            max_results: Maximum papers to return
        
        Returns:
            List of paper details
        """
        # Enhance query for better results
        query = f"{compound_name}[Title/Abstract]"
        
        pmids = self.search(query, max_results)
        
        if pmids:
            return self.fetch_details(pmids)
        return []
    
    def get_recent_papers(self, query: str, years: int = 5, max_results: int = 10) -> List[Dict]:
        """
        Get recent papers from last N years.
        
        Args:
            query: Search term
            years: Number of recent years
            max_results: Maximum results
        
        Returns:
            List of recent papers
        """
        current_year = datetime.now().year
        year_filter = f"{current_year - years}:{current_year}[PDAT]"
        
        enhanced_query = f"{query} AND {year_filter}"
        
        pmids = self.search(enhanced_query, max_results)
        
        if pmids:
            return self.fetch_details(pmids)
        return []
    
    def summarize_abstract(self, abstract: str, max_sentences: int = 3) -> str:
        """
        Create a simple summary of abstract.
        
        For now, returns first N sentences. Can be enhanced with
        transformers library for AI summarization.
        
        Args:
            abstract: Full abstract text
            max_sentences: Number of sentences for summary
        
        Returns:
            Summary text
        """
        if not abstract or len(abstract) < 100:
            return abstract
        
        # Simple sentence splitting
        sentences = abstract.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        
        # Return first N sentences
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences)
        
        # Add ellipsis if truncated
        if len(sentences) > max_sentences:
            summary += '...'
        
        return summary
    
    def export_bibtex(self, paper: Dict) -> str:
        """
        Export paper citation as BibTeX format.
        
        Args:
            paper: Paper dictionary
        
        Returns:
            BibTeX citation string
        """
        authors_str = ' and '.join(paper.get('authors', []))
        
        bibtex = f"""@article{{pmid{paper['pmid']},
    author = {{{authors_str}}},
    title = {{{paper['title']}}},
    journal = {{{paper['journal']}}},
    year = {{{paper['year']}}},
    pmid = {{{paper['pmid']}}},
    url = {{{paper['url']}}}
}}"""
        
        if paper.get('doi'):
            bibtex = bibtex.replace('}}}', f"}},\n    doi = {{{paper['doi']}}}\n}}")
        
        return bibtex


def search_literature(molecule_name: str, max_results: int = 10) -> List[Dict]:
    """
    Convenience function to search literature for a molecule.
    
    Args:
        molecule_name: Compound/drug name
        max_results: Maximum papers to return
    
    Returns:
        List of paper details with summaries
    """
    api = PubMedAPI()
    papers = api.search_by_compound(molecule_name, max_results)
    
    # Add summaries
    for paper in papers:
        paper['summary'] = api.summarize_abstract(paper['abstract'])
    
    return papers
