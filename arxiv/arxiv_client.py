import feedparser
import requests
import os

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def search_arxiv(query, max_results=5):
    search_query = f"all:{query.replace(' ', '+')}"
    url = f"{ARXIV_API_URL}?search_query={search_query}&start=0&max_results={max_results}"

    feed = feedparser.parse(url)

    results = []
    for entry in feed.entries:
        pdf_link = None
        for link in entry.links:
            if link.type == "application/pdf":
                pdf_link = link.href
                break

        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": pdf_link
        })

    return results


def download_pdf(pdf_url, save_path):
    try:
        # arXiv requires a custom User-Agent to avoid 403s
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(pdf_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return False
