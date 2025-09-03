"""
Biotech News Agent (free) — Python single-file agent

What it does:
 - Fetches biotech-related news from multiple RSS feeds
 - Filters and ranks important items by keyword + recency
 - Summarizes each article (two modes: OpenAI API or local HF summarizer)
 - Generates a LinkedIn-ready post (text + links)
 - Optionally posts to LinkedIn if you provide an ACCESS_TOKEN and AUTHOR urn

How to use:
 1. Install dependencies:
    pip install -r requirements.txt
    (requirements.txt content is listed below in comments)

 2. Configure env vars or edit CONFIG section in the file:
    - OPENAI_API_KEY (optional) for better summaries
    - LINKEDIN_ACCESS_TOKEN (optional) for auto-posting
    - LINKEDIN_AUTHOR_URN (optional) e.g. "urn:li:person:xxxx" or page urn

 3. Run once: python biotech_news_agent.py
    - It will create ./out/ with summaries and the LinkedIn post text

 4. To run daily: add a cron job or use GitHub Actions. See the bottom of this file for a sample GitHub Actions yaml snippet.

Notes:
 - This agent is intentionally simple and depends on freely available RSS feeds.
 - If you want automatic posting, create a LinkedIn app and obtain an access token and author URN.
 - If you don't have OPENAI_API_KEY, the script will try a local HF summarizer (slow on first run and requires model download).

Requirements (put into requirements.txt):
feedparser
requests
beautifulsoup4
python-dotenv
transformers[sentencepiece]>=4.0.0
torch>=1.12  # optional but recommended for HF summarizer
newspaper3k  # optional for article text extraction
tqdm

(If you prefer only OpenAI summarization, you can omit transformers and torch.)

"""

import os
import re
import json
import time
import math
from datetime import datetime, timedelta
from collections import defaultdict

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urlparse

# Optional imports for summarization
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Optional OpenAI import; we use direct requests to the OpenAI API to avoid pinning a client lib.

# ---------------- CONFIG ----------------
# You can set these via environment variables or edit here.
CONFIG = {
    "RSS_FEEDS": [
        # general science/biotech feeds
        "https://www.nature.com/nature/articles?type=article&format=rss",
        "https://www.sciencemag.org/rss/news.xml",
        "https://www.sciencedaily.com/rss/health_medicine/biotechnology.xml",
        "https://www.fiercebiotech.com/rss.xml",
        "https://www.genomeweb.com/rss.xml",
        "https://www.biorxiv.org/rss/latest.xml",
        "https://www.nih.gov/news-events/news-releases/feed",
        # add more RSS URLs here as desired
    ],
    "KEYWORDS": [
        # keywords to boost importance (case-insensitive)
        "CRISPR", "gene therapy", "mRNA", "vaccine", "clinical trial", "FDA", "biosimilar",
        "biotech", "synthetic biology", "CAR-T", "genome", "sequencing", "bioinformatics",
    ],
    "MAX_ARTICLES": 10,  # how many top items to include in the LinkedIn post
    "TIME_WINDOW_HOURS": 48,  # only consider articles published within this many hours
    "OUTPUT_DIR": "out",
    "USE_OPENAI": bool(os.getenv("OPENAI_API_KEY")),
    # LinkedIn posting (optional): set these environment variables or edit here
    "LINKEDIN_ACCESS_TOKEN": os.getenv("LINKEDIN_ACCESS_TOKEN"),
    "LINKEDIN_AUTHOR_URN": os.getenv("LINKEDIN_AUTHOR_URN"),
}

# ---------------- Helpers ----------------

def safe_filename(s):
    return re.sub(r"[^0-9a-zA-Z-_\. ]+", "_", s)[:200]


def fetch_feed(url):
    try:
        return feedparser.parse(url)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


def extract_text_from_url(url):
    # Try to get main text using BeautifulSoup; newspaper3k would be better but optional.
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; BiotechNewsAgent/1.0)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        # Heuristic: find article tag, or largest <p> collection
        article = soup.find("article")
        if article:
            text = "\n".join(p.get_text(strip=True) for p in article.find_all("p"))
        else:
            # fallback: join all <p>
            ps = soup.find_all("p")
            text = "\n".join(p.get_text(strip=True) for p in ps[:60])
        # truncate
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"Could not extract text from {url}: {e}")
        return ""


def score_article(title, summary, keywords=CONFIG['KEYWORDS']):
    s = 0
    text = (title + " " + (summary or "")).lower()
    for k in keywords:
        if k.lower() in text:
            s += 2
    # boost if title contains high-impact words
    for w, v in [("study", 1), ("trial", 2), ("approval", 3), ("FDA", 3), ("breakthrough", 2)]:
        if w.lower() in text:
            s += v
    return s

# ---------------- Summarization ----------------

class Summarizer:
    def __init__(self):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_summarizer = None
        if not self.openai_key and HF_AVAILABLE:
            # create a small summarizer pipeline (first call will download model)
            try:
                self.hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            except Exception as e:
                print("Could not initialize HF summarizer:", e)
                self.hf_summarizer = None

    def summarize(self, text, max_chars=600):
        if not text:
            return ""
        text = text.strip()
        if self.openai_key:
            return self._openai_summarize(text, max_chars=max_chars)
        elif self.hf_summarizer:
            return self._hf_summarize(text, max_chars=max_chars)
        else:
            # fallback: tiny extractive summary: first 2 sentences
            s = re.split(r"(?<=[.!?])\s+", text)
            return " ".join(s[:2])[:max_chars]

    def _openai_summarize(self, text, max_chars=600):
        # Use OpenAI ChatCompletions via REST. This expects OPENAI_API_KEY is set.
        prompt = (
            "Summarize the following article in 2-3 concise sentences suitable for a LinkedIn post. "
            "Keep it neutral, mention one key finding or news point, and provide the source link at the end.\n\nArticle:\n" + text
        )
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-4o-mini",  # common available model name may vary; user can change
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.2,
        }
        try:
            r = requests.post(url, headers=headers, json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            content = data['choices'][0]['message']['content'].strip()
            return content[:max_chars]
        except Exception as e:
            print("OpenAI summarization failed:", e)
            return self._hf_summarize(text, max_chars) if self.hf_summarizer else (text[:max_chars])

    def _hf_summarize(self, text, max_chars=600):
        try:
            # HF summarizer expects not-too-long text; we truncate intelligently
            prefix = text[:8000]
            out = self.hf_summarizer(prefix, max_length=130, min_length=30, do_sample=False)
            s = out[0]['summary_text']
            return s[:max_chars]
        except Exception as e:
            print("HF summarization failed:", e)
            s = re.split(r"(?<=[.!?])\s+", text)
            return " ".join(s[:2])[:max_chars]

# ---------------- LinkedIn Posting ----------------

def build_linkedin_post(items, max_chars_total=1200):
    # items: list of dicts with keys: title, summary, link, score, source
    header = "Daily biotech news roundup — " + datetime.utcnow().strftime("%Y-%m-%d") + "\n\n"
    lines = [header]
    for i, it in enumerate(items[:CONFIG['MAX_ARTICLES']]):
        # Format: • Title (Source) — one-sentence summary. Link
        title = re.sub(r"\s+", " ", it['title']).strip()
        summary = re.sub(r"\s+", " ", it['summary']).strip()
        src = it.get('source') or urlparse(it['link']).netloc
        line = f"• {title} ({src}) — {summary} {it['link']}"
        lines.append(line)
    post = "\n\n".join(lines)
    if len(post) > max_chars_total:
        # truncate more aggressively: keep only top 5 items
        post = "Daily biotech news roundup — " + datetime.utcnow().strftime("%Y-%m-%d") + "\n\n"
        for it in items[:5]:
            post += f"• {it['title']} — {it['summary']} {it['link']}\n\n"
        post = post[:max_chars_total]
    return post


def post_to_linkedin(access_token, author_urn, text):
    # LinkedIn v2 text post endpoint
    # Documentation: https://docs.microsoft.com/en-us/linkedin/marketing/integrations/community-management/shares/posts-api
    api_url = "https://api.linkedin.com/v2/ugcPosts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    body = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
    }
    r = requests.post(api_url, headers=headers, json=body, timeout=20)
    if r.status_code in (201, 200):
        return True, r.json()
    else:
        return False, r.text

# ---------------- Main flow ----------------


def run_agent():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    summarizer = Summarizer()

    cutoff = datetime.utcnow() - timedelta(hours=CONFIG['TIME_WINDOW_HOURS'])
    collected = []
    seen_urls = set()

    print("Fetching feeds...")
    for feed_url in CONFIG['RSS_FEEDS']:
        print("  ->", feed_url)
        fd = fetch_feed(feed_url)
        if not fd or 'entries' not in fd:
            continue
        for entry in fd['entries']:
            # entries vary; try to extract link, title, published
            link = entry.get('link') or entry.get('id')
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            title = entry.get('title', '')
            summary = entry.get('summary', '')
            # published time
            pub_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
            if pub_parsed:
                pub_dt = datetime.utcfromtimestamp(time.mktime(pub_parsed))
            else:
                pub_dt = datetime.utcnow()
            if pub_dt < cutoff:
                continue
            source = entry.get('source', {}).get('title') if entry.get('source') else None
            collected.append({
                'title': title,
                'summary_raw': BeautifulSoup(summary, 'html.parser').get_text() if summary else '',
                'link': link,
                'published': pub_dt,
                'source': source,
            })

    print(f"Collected {len(collected)} recent entries")

    # For each, extract text & summarize & score
    enriched = []
    for i, c in enumerate(sorted(collected, key=lambda x: x['published'], reverse=True)):
        print(f"Processing {i+1}/{len(collected)}: {c['title'][:80]}")
        full_text = extract_text_from_url(c['link'])
        if not full_text:
            full_text = c['summary_raw'] or c['title']
        summary = summarizer.summarize(full_text, max_chars=280)
        score = score_article(c['title'], summary)
        enriched.append({
            'title': c['title'],
            'link': c['link'],
            'published': c['published'].isoformat(),
            'summary': summary,
            'score': score,
            'source': c.get('source') or urlparse(c['link']).netloc,
        })

    # Rank by score then recency
    enriched.sort(key=lambda x: (x['score'], x['published']), reverse=True)

    # Save JSON for records
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(CONFIG['OUTPUT_DIR'], f"biotech_news_{ts}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print("Saved raw results to", json_path)

    # Build LinkedIn post text
    top_items = enriched[:CONFIG['MAX_ARTICLES']]
    post_text = build_linkedin_post(top_items)
    txt_path = os.path.join(CONFIG['OUTPUT_DIR'], f"linkedin_post_{ts}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(post_text)
    print("Prepared LinkedIn post at", txt_path)

    # Optionally, auto-post
    if CONFIG['LINKEDIN_ACCESS_TOKEN'] and CONFIG['LINKEDIN_AUTHOR_URN']:
        print("Posting to LinkedIn...")
        ok, resp = post_to_linkedin(CONFIG['LINKEDIN_ACCESS_TOKEN'], CONFIG['LINKEDIN_AUTHOR_URN'], post_text)
        if ok:
            print("Posted successfully. Response:", resp)
        else:
            print("Failed to post to LinkedIn:", resp)
    else:
        print("No LinkedIn credentials provided. Open the text file and post manually if you like.")

    print("Done.")


if __name__ == '__main__':
    run_agent()

# ---------------- Extra: GitHub Actions example ----------------
# Save the following as .github/workflows/daily-post.yml to run daily on GitHub (you must provide secrets):
#
# name: Daily Biotech News
# on:
#   schedule:
#     - cron: '0 7 * * *'  # daily at 07:00 UTC (adjust as needed)
# jobs:
#   build:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v3
#       - name: Set up Python
#         uses: actions/setup-python@v4
#         with:
#           python-version: '3.10'
#       - name: Install deps
#         run: |
#           python -m pip install --upgrade pip
#           pip install feedparser requests beautifulsoup4 python-dotenv transformers torch newspaper3k
#       - name: Run agent
#         env:
#           OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
#           LINKEDIN_ACCESS_TOKEN: ${{ secrets.LINKEDIN_ACCESS_TOKEN }}
#           LINKEDIN_AUTHOR_URN: ${{ secrets.LINKEDIN_AUTHOR_URN }}
#         run: |
#           python biotech_news_agent.py
#
# Make sure to add any required secrets to your GitHub repository.
