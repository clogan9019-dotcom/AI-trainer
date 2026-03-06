#!/usr/bin/env python3
"""
LLM Training CLI Tool with Auto Mode
"""

import click
import json
import re
import hashlib
import gzip
import os
from pathlib import Path
from typing import List, Set, Optional, Dict, Generator
import requests
from bs4 import BeautifulSoup
import time
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from datetime import datetime, timedelta
import random
from urllib.parse import urlparse
import warnings
import math

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
except Exception:
    Tokenizer = None

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "a", "is", "it", "that", "for", "on",
    "with", "as", "are", "was", "be", "by", "this", "from", "at", "an", "which",
    "have", "has", "were", "not", "but", "their", "its", "they", "he", "she",
    "we", "you", "his", "her", "them", "there", "what", "when", "where", "who",
    "how", "why", "can", "could", "would", "should", "will", "may", "might"
}


WIKI_PATH_BLACKLIST = (
    "/wiki/Special:",
    "/wiki/File:",
    "/wiki/Help:",
    "/wiki/Talk:",
    "/wiki/Category:",
    "/wiki/Portal:",
    "/wiki/Template:",
    "/wiki/User:",
    "/wiki/Wikipedia:",
)


def is_blacklisted_wiki_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = parsed.netloc.lower()
    if not host.endswith("wikipedia.org"):
        return False
    path = parsed.path or ""
    for prefix in WIKI_PATH_BLACKLIST:
        if path.startswith(prefix):
            return True
    return False


def load_seen_urls(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                seen.add(url)
    return seen


def append_seen_url(path: Path, url: str) -> None:
    if not url:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(url + "\n")


class ImprovedCommonCrawlFetcher:
    """Fetch data from Common Crawl"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cc_indexes = self._refresh_indexes() or [
            "CC-MAIN-2025-38",
            "CC-MAIN-2024-10",
            "CC-MAIN-2023-50",
            "CC-MAIN-2023-40",
        ]
        
        self.index_base = "https://index.commoncrawl.org"
        self.s3_base = "https://data.commoncrawl.org"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational LLM Trainer)'
        })

    def _refresh_indexes(self) -> List[str]:
        try:
            resp = requests.get("https://index.commoncrawl.org/collinfo.json", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            data.sort(key=lambda x: x.get("id", ""), reverse=True)
            return [row.get("id") for row in data if row.get("id")]
        except Exception:
            return []
    
    def _cc_index_query(self, url_pattern: str, index: str,
                        filter_pattern: str = None,
                        match_type: str = "domain",
                        limit: int = 100,
                        retries: int = 5) -> List[Dict]:
        url = f"{self.index_base}/{index}-index"
        
        params = {
            'url': url_pattern,
            'matchType': match_type,
            'output': 'json',
        }
        
        if filter_pattern:
            params['filter'] = filter_pattern
        
        last_err = None
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=60)
                if response.status_code != 200:
                    last_err = RuntimeError(f"HTTP {response.status_code}")
                    time.sleep(min(2 ** attempt, 30))
                    continue
                results = []
                for line in response.text.strip().split('\n'):
                    if line and len(results) < limit:
                        try:
                            record = json.loads(line)
                            results.append(record)
                        except json.JSONDecodeError:
                            continue
                return results
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        if last_err:
            return []
    
    def _common_crawl_stream(self, filename: str, offset: int, length: int) -> Optional[bytes]:
        url = f"{self.s3_base}/{filename}"
        headers = {'Range': f'bytes={offset}-{offset + length - 1}'}
        
        try:
            response = self.session.get(url, headers=headers, timeout=60, stream=True)
            if response.status_code not in [200, 206]:
                return None
            return response.content
        except Exception:
            return None
    
    def _warc_extract_text(self, warc_data: bytes) -> Optional[str]:
        try:
            try:
                decompressed = gzip.decompress(warc_data)
            except Exception:
                decompressed = warc_data
            
            content = decompressed.decode('utf-8', errors='ignore')
            parts = content.split('\r\n\r\n', 2)
            
            if len(parts) < 3:
                parts = content.split('\n\n', 2)
            
            if len(parts) < 3:
                return None
            
            body = parts[2] if len(parts) > 2 else ""
            
            if body.startswith('\r\n'):
                body = body[2:]
            elif body.startswith('\n'):
                body = body[1:]
            
            return body
        except Exception:
            return None
    
    def _extract_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header',
                                'aside', 'form', 'button', 'iframe', 'noscript',
                                'svg', 'path', 'meta', 'link']):
                element.decompose()
            
            # Remove Wikipedia specific junk
            for element in soup.find_all('div', {'class': re.compile(r'interlanguage|sidebar|navbox|catlinks|mw-jump|mw-navigation|footer', re.I)}):
                element.decompose()
            
            for element in soup.find_all('ul', {'class': re.compile(r'interlanguage', re.I)}):
                element.decompose()
            
            for element in soup.find_all('li', {'class': re.compile(r'interlanguage|interwiki', re.I)}):
                element.decompose()
            
            # Find main content
            main_content = (
                soup.find('div', {'id': 'mw-content-text'}) or
                soup.find('article') or 
                soup.find('main') or 
                soup.body or
                soup
            )
            
            text = main_content.get_text(separator=' ', strip=True)
            
            # Remove non-ASCII (foreign language text)
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            
            # Clean up
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
            text = re.sub(r'http\S+', '', text)
            
            # Remove very short remaining text
            if len(text) < 500:
                return ""
            
            return text.strip()
        except Exception:
            return ""
    
    def fetch_from_index(self, url_pattern: str = "wikipedia.org",
                         index: str = None,
                         max_records: int = 50,
                         match_type: str = "domain",
                         filter_pattern: str = "status:200") -> Generator[Dict, None, None]:
        if index is None:
            index = self.cc_indexes[0]
        
        records = self._cc_index_query(
            url_pattern, 
            index, 
            filter_pattern=filter_pattern,
            match_type=match_type,
            limit=max_records * 2
        )
        
        if not records:
            return
        
        fetched = 0
        for record in records:
            if fetched >= max_records:
                break
            
            filename = record.get('filename')
            offset = int(record.get('offset', 0))
            length = int(record.get('length', 0))
            url = record.get('url', '')
            timestamp = record.get('timestamp', '')
            
            if not filename or length == 0:
                continue
            
            warc_data = self._common_crawl_stream(filename, offset, length)
            if not warc_data:
                continue
            
            html = self._warc_extract_text(warc_data)
            if not html or len(html) < 100:
                continue
            
            text = self._extract_text(html)
            if not text or len(text) < 500:
                continue
            
            yield {
                'url': url,
                'text': text,
                'source': 'common_crawl',
                'timestamp': timestamp,
                'index': index
            }
            
            fetched += 1
            time.sleep(0.5)
    
    def fetch_multiple_domains(self, domains: List[str] = None, 
                               records_per_domain: int = 20) -> List[Dict]:
        if domains is None:
            domains = ["wikipedia.org"]
        
        all_records = []
        
        for domain in domains:
            click.echo(f"\nFetching from: {domain}")
            
            domain_records = []
            for index in self.cc_indexes:
                if len(domain_records) >= records_per_domain:
                    break
                
                try:
                    for record in self.fetch_from_index(
                        url_pattern=domain,
                        index=index,
                        max_records=records_per_domain - len(domain_records)
                    ):
                        domain_records.append(record)
                        if len(domain_records) >= records_per_domain:
                            break
                    
                    if domain_records:
                        break
                except Exception:
                    continue
            
            all_records.extend(domain_records)
            click.echo(f"  Got {len(domain_records)} records")
        
        return all_records


class MediaWikiFetcher:
    """Fetch data from MediaWiki API (Wikipedia)"""

    def __init__(self, api_url: str = "https://en.wikipedia.org/w/api.php"):
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational LLM Trainer)'
        })

    def _get(self, params: dict, retries: int = 3) -> dict:
        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.get(self.api_url, params=params, timeout=30)
                if resp.status_code in (429, 503):
                    time.sleep(min(2 ** attempt, 10))
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        if last_err:
            raise last_err
        return {}

    def fetch_random(self, count: int = 50) -> List[Dict]:
        records = []
        if count <= 0:
            return records

        # MediaWiki API caps random list at 50
        remaining = count
        while remaining > 0:
            batch = min(50, remaining)
            params = {
                "action": "query",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": batch,
                "format": "json",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("random", [])
            pageids = [str(p["id"]) for p in pages if "id" in p]
            if not pageids:
                break

            # Fetch extracts in one call
            params = {
                "action": "query",
                "prop": "extracts",
                "pageids": "|".join(pageids),
                "explaintext": 1,
                "format": "json",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                title = page.get("title", "")
                text = page.get("extract", "") or ""
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
                if text:
                    records.append({
                        "url": url,
                        "text": text,
                        "source": "mediawiki",
                        "timestamp": datetime.now().isoformat()
                    })
            remaining = count - len(records)

        return records

    def fetch_from_categories(self, categories: List[str], count: int = 50) -> List[Dict]:
        records = []
        if not categories or count <= 0:
            return records

        remaining = count
        cat_index = 0
        while remaining > 0:
            category = categories[cat_index % len(categories)]
            cat_index += 1

            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmnamespace": 0,
                "cmlimit": min(50, remaining),
                "format": "json",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("categorymembers", [])
            pageids = [str(p["pageid"]) for p in pages if "pageid" in p]
            if not pageids:
                continue

            params = {
                "action": "query",
                "prop": "extracts",
                "pageids": "|".join(pageids),
                "explaintext": 1,
                "format": "json",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                title = page.get("title", "")
                text = page.get("extract", "") or ""
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
                if text:
                    records.append({
                        "url": url,
                        "text": text,
                        "source": "mediawiki",
                        "timestamp": datetime.now().isoformat()
                    })
            remaining = count - len(records)

        return records

    def fetch_by_titles(self, titles: List[str]) -> List[Dict]:
        records = []
        if not titles:
            return records
        # MediaWiki titles parameter supports batching
        for i in range(0, len(titles), 50):
            batch = titles[i:i + 50]
            params = {
                "action": "query",
                "prop": "extracts",
                "titles": "|".join(batch),
                "explaintext": 1,
                "format": "json",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                title = page.get("title", "")
                text = page.get("extract", "") or ""
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
                if text:
                    records.append({
                        "url": url,
                        "text": text,
                        "source": "mediawiki",
                        "timestamp": datetime.now().isoformat()
                    })
        return records

    def fetch_top_pageviews(self, count: int = 50, days_back: int = 1) -> List[Dict]:
        # Wikimedia Pageviews API
        target = datetime.utcnow() - timedelta(days=days_back)
        year = target.strftime("%Y")
        month = target.strftime("%m")
        day = target.strftime("%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia.org/all-access/{year}/{month}/{day}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return []
        articles = items[0].get("articles", [])
        titles = []
        for a in articles:
            title = a.get("article", "")
            if not title or title.startswith("Special:"):
                continue
            if title == "Main_Page":
                continue
            titles.append(title.replace("_", " "))
            if len(titles) >= count:
                break
        return self.fetch_by_titles(titles)


class StackExchangeFetcher:
    """Fetch data from Stack Exchange API (Stack Overflow)."""

    def __init__(self, site: str = "stackoverflow", api_key: Optional[str] = None):
        self.site = site
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational LLM Trainer)'
        })

    def _get(self, url: str, params: dict) -> dict:
        base = {
            "site": self.site,
        }
        if self.api_key:
            base["key"] = self.api_key
        base.update(params)
        resp = self.session.get(url, params=base, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_python(self, count: int = 50) -> List[Dict]:
        records = []
        if count <= 0:
            return records
        page = 1
        while len(records) < count:
            data = self._get("https://api.stackexchange.com/2.3/questions", {
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
                "pagesize": 30,
                "tagged": "python",
                "page": page,
            })
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                title = item.get("title", "")
                body = item.get("body", "")
                if not body:
                    continue
                text = f"{title}\n{body}"
                records.append({
                    "url": item.get("link", ""),
                    "text": text,
                    "source": "stackexchange",
                    "timestamp": datetime.now().isoformat()
                })
                if len(records) >= count:
                    break
            if not data.get("has_more"):
                break
            page += 1
        return records

    def fetch_python_qa(self, count: int = 50) -> List[Dict]:
        records = []
        if count <= 0:
            return records
        page = 1
        while len(records) < count:
            qdata = self._get("https://api.stackexchange.com/2.3/questions", {
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
                "pagesize": 30,
                "tagged": "python",
                "page": page,
            })
            items = qdata.get("items", [])
            if not items:
                break
            for q in items:
                qid = q.get("question_id")
                if not qid:
                    continue
                q_title = q.get("title", "")
                q_body = q.get("body", "")
                if not q_body:
                    continue

                adata = self._get(f"https://api.stackexchange.com/2.3/questions/{qid}/answers", {
                    "order": "desc",
                    "sort": "votes",
                    "filter": "withbody",
                    "pagesize": 1,
                })
                answers = adata.get("items", [])
                if not answers:
                    continue
                a_body = answers[0].get("body", "")
                if not a_body:
                    continue

                text = f"User: {q_title}\n{q_body}\nAssistant: {a_body}"
                records.append({
                    "url": q.get("link", ""),
                    "text": text,
                    "source": "stackexchange_qa",
                    "timestamp": datetime.now().isoformat()
                })
                if len(records) >= count:
                    break

            if not qdata.get("has_more"):
                break
            page += 1
        return records


class FallbackDataSources:
    """Generate training data when Common Crawl fails"""
    
    def __init__(self):
        self.topics = [
            {
                'subject': 'artificial intelligence',
                'facts': [
                    'Artificial intelligence is transforming how we live and work in the modern world.',
                    'Machine learning algorithms can identify patterns in large datasets.',
                    'Neural networks are inspired by the structure of the human brain.',
                    'Deep learning has enabled breakthroughs in image and speech recognition.',
                    'AI systems can now beat humans at complex games like chess and Go.',
                    'Natural language processing allows computers to understand human speech.',
                    'Robotics combines AI with mechanical engineering to create intelligent machines.',
                    'Computer vision enables machines to interpret and understand visual information.',
                ]
            },
            {
                'subject': 'biology',
                'facts': [
                    'Cells are the fundamental units of all living organisms on Earth.',
                    'DNA contains the genetic instructions for development and function.',
                    'Photosynthesis converts sunlight into chemical energy in plants.',
                    'Evolution occurs through natural selection over many generations.',
                    'The human body contains trillions of cells working together.',
                    'Ecosystems maintain balance through complex food webs and interactions.',
                    'Mitosis is the process by which cells divide and reproduce.',
                    'Proteins are essential molecules that perform most cellular functions.',
                ]
            },
            {
                'subject': 'physics',
                'facts': [
                    'Gravity is the force that attracts objects with mass toward each other.',
                    'Light travels at approximately 300000 kilometers per second.',
                    'Energy cannot be created or destroyed only transformed.',
                    'Atoms are composed of protons neutrons and electrons.',
                    'Quantum mechanics describes behavior at the subatomic level.',
                    'Electricity is the flow of electrons through a conductor.',
                    'Magnetism and electricity are related phenomena.',
                    'The laws of thermodynamics govern energy transfer in systems.',
                ]
            },
            {
                'subject': 'chemistry',
                'facts': [
                    'Elements are organized in the periodic table by atomic number.',
                    'Chemical reactions involve the breaking and forming of bonds.',
                    'Water is a molecule composed of two hydrogen atoms and one oxygen atom.',
                    'Acids and bases react together in neutralization reactions.',
                    'Carbon forms the basis of organic chemistry and life itself.',
                    'Metals conduct electricity because of their electron structure.',
                    'Chemical equilibrium occurs when reaction rates are balanced.',
                    'Catalysts speed up reactions without being consumed.',
                ]
            },
            {
                'subject': 'history',
                'facts': [
                    'Ancient civilizations developed writing systems thousands of years ago.',
                    'The Renaissance was a period of cultural rebirth in Europe.',
                    'The Industrial Revolution transformed manufacturing and society.',
                    'World War Two was the deadliest conflict in human history.',
                    'The printing press revolutionized the spread of information.',
                    'Ancient Egypt built the pyramids as tombs for pharaohs.',
                    'The Roman Empire influenced law government and architecture.',
                    'The Scientific Revolution changed how humans understand nature.',
                ]
            },
            {
                'subject': 'geography',
                'facts': [
                    'The Earth has seven continents and five major oceans.',
                    'Mountains form when tectonic plates collide over millions of years.',
                    'Rivers carry water from highlands to the sea.',
                    'Climate varies based on latitude altitude and proximity to water.',
                    'Volcanoes release molten rock from deep within the Earth.',
                    'Deserts receive very little rainfall throughout the year.',
                    'The water cycle moves water between land sea and atmosphere.',
                    'Earthquakes occur along fault lines in the Earth crust.',
                ]
            },
            {
                'subject': 'astronomy',
                'facts': [
                    'The solar system contains eight planets orbiting the Sun.',
                    'Stars are massive balls of hot gas undergoing nuclear fusion.',
                    'Galaxies contain billions of stars held together by gravity.',
                    'The Moon orbits the Earth once every twenty eight days.',
                    'Black holes have gravity so strong that light cannot escape.',
                    'The universe is approximately fourteen billion years old.',
                    'Planets form from disks of gas and dust around young stars.',
                    'Comets are icy bodies that develop tails when near the Sun.',
                ]
            },
            {
                'subject': 'mathematics',
                'facts': [
                    'Numbers can be classified as natural integers rational or irrational.',
                    'Geometry studies shapes sizes and properties of space.',
                    'Algebra uses symbols to represent numbers in equations.',
                    'Calculus deals with rates of change and accumulation.',
                    'Statistics helps us analyze and interpret data.',
                    'Prime numbers are only divisible by one and themselves.',
                    'The Pythagorean theorem relates sides of right triangles.',
                    'Probability measures the likelihood of events occurring.',
                ]
            },
        ]
        
        self.connectors = [
            'Furthermore',
            'Additionally', 
            'Moreover',
            'In addition',
            'Research shows that',
            'Scientists have found that',
            'Studies indicate that',
            'It is well established that',
            'Experts agree that',
            'Evidence suggests that',
        ]
        
        self.conclusions = [
            'This knowledge continues to shape our understanding of the world.',
            'Researchers continue to explore these fascinating topics.',
            'Understanding these concepts is essential for scientific literacy.',
            'These principles have many practical applications in daily life.',
            'Modern technology has enabled new discoveries in this field.',
            'Students around the world study these important subjects.',
            'The applications of this knowledge benefit society greatly.',
            'Ongoing research reveals new insights every year.',
        ]
    
    def generate_texts(self, count: int) -> List[str]:
        """Generate long unique training texts"""
        texts = []
        
        for i in range(count):
            num_topics = random.randint(2, 3)
            selected_topics = random.sample(self.topics, num_topics)
            
            paragraphs = []
            
            for topic in selected_topics:
                num_facts = random.randint(3, 5)
                facts = random.sample(topic['facts'], min(num_facts, len(topic['facts'])))
                
                paragraph = f"The study of {topic['subject']} is important in modern education. "
                
                for j, fact in enumerate(facts):
                    if j > 0:
                        connector = random.choice(self.connectors)
                        paragraph += f" {connector} {fact.lower()}"
                    else:
                        paragraph += fact
                
                paragraph += " " + random.choice(self.conclusions)
                paragraphs.append(paragraph)
            
            full_text = " ".join(paragraphs)
            unique_id = f" Reference number {i + random.randint(1000, 9999)}."
            full_text += unique_id
            
            texts.append(full_text)
        
        for i in range(count):
            topic = random.choice(self.topics)
            text = f"An introduction to {topic['subject']}. "
            
            for fact in topic['facts']:
                text += fact + " "
            
            topic2 = random.choice([t for t in self.topics if t != topic])
            text += f"Related concepts in {topic2['subject']} include the following. "
            
            for fact in topic2['facts']:
                text += fact + " "
            
            text += random.choice(self.conclusions)
            text += f" Document {i + random.randint(10000, 99999)}."
            
            texts.append(text)
        
        return texts[:count]


class DataSourceManager:
    """Manage data fetching"""
    
    def __init__(self, output_dir: str = "data/raw", config: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cc_fetcher = ImprovedCommonCrawlFetcher(output_dir)
        cfg = config or {}
        mw_urls = cfg.get("mediawiki_endpoints") or [cfg.get("mediawiki_api", "https://en.wikipedia.org/w/api.php")]
        self.mw_fetchers = [MediaWikiFetcher(api_url=u) for u in mw_urls if u]
        self.se_fetcher = StackExchangeFetcher(
            site=cfg.get("stackexchange_site", "stackoverflow"),
            api_key=cfg.get("stackexchange_key")
        )
        self.fallback = FallbackDataSources()
        self.consecutive_failures = 0
        self.seen_urls_path = self.output_dir / "seen_urls.txt"
        self.seen_urls = load_seen_urls(self.seen_urls_path)
        self.seen_titles_path = self.output_dir / "seen_titles.txt"
        self.seen_titles = load_seen_urls(self.seen_titles_path)
        self.config = config or {}
    
    def fetch_data(self, count: int = 50, use_fallback: bool = True) -> List[Dict]:
        """Fetch training data"""
        records_out = []
        
        try:
            use_stackexchange = self.config.get("use_stackexchange", True)
            use_mediawiki = self.config.get("use_mediawiki", True)
            use_commoncrawl = self.config.get("use_commoncrawl", False)
            records = []

            if use_stackexchange:
                click.echo("\nFetching from Stack Exchange API (Python training data)...")
                se_count = self.config.get("stackexchange_docs", max(10, count // 2))
                se_qa = self.config.get("stackexchange_qa", True)
                try:
                    if se_qa:
                        records.extend(self.se_fetcher.fetch_python_qa(se_count))
                    else:
                        records.extend(self.se_fetcher.fetch_python(se_count))
                except Exception:
                    pass
                click.echo(f"  Records total so far: {len(records)}")

            if use_mediawiki and len(records) < count:
                click.echo("\nFetching from MediaWiki API (general training data)...")
                use_pageviews = self.config.get("use_pageviews", True)
                for fetcher in self.mw_fetchers:
                    if len(records) >= count:
                        break
                    click.echo(f"  Trying endpoint: {fetcher.api_url}")
                    if use_pageviews:
                        days_back = self.config.get("pageviews_days_back", 1)
                        try:
                            more = fetcher.fetch_top_pageviews(count - len(records), days_back=days_back)
                            records.extend(more)
                        except Exception:
                            pass
                    if len(records) < count:
                        categories = self.config.get("mediawiki_categories", [])
                        try:
                            if categories:
                                more = fetcher.fetch_from_categories(categories, count - len(records))
                                records.extend(more)
                            else:
                                more = fetcher.fetch_random(count - len(records))
                                records.extend(more)
                        except Exception:
                            pass
                    click.echo(f"  Records total so far: {len(records)}")

            if use_commoncrawl and len(records) < count:
                click.echo("\nFetching from Common Crawl (general training data)...")
                url_pattern = self.config.get("url_pattern")
                match_type = self.config.get("match_type", "domain")
                cc_index = self.config.get("cc_index") or None
                cc_filter = self.config.get("cc_filter", "status:200")
                domains = self.config.get("domains", ["wikipedia.org"])

                if url_pattern:
                    records = list(self.cc_fetcher.fetch_from_index(
                        url_pattern=url_pattern,
                        index=cc_index,
                        max_records=count,
                        match_type=match_type,
                        filter_pattern=cc_filter
                    ))
                else:
                    records = self.cc_fetcher.fetch_multiple_domains(
                        domains=domains,
                        records_per_domain=count
                    )
            
            for record in records:
                url = record.get("url", "")
                title = ""
                if url and "wikipedia.org/wiki/" in url:
                    title = url.split("/wiki/")[-1]
                    title = title.replace("_", " ").strip().lower()
                if title and title in self.seen_titles:
                    continue
                if url and url in self.seen_urls:
                    continue
                if url:
                    self.seen_urls.add(url)
                    append_seen_url(self.seen_urls_path, url)
                if title:
                    self.seen_titles.add(title)
                    append_seen_url(self.seen_titles_path, title)
                records_out.append(record)
            
            if len(records_out) > 0:
                click.echo(f"Fetched {len(records_out)} documents (after dedupe)")
            else:
                if records:
                    click.echo(f"No new documents after dedupe ({len(records)} fetched)")
                else:
                    click.echo("No records returned")
        except Exception as e:
            click.echo(f"Data source error: {str(e)[:50]}")
        
        if len(records_out) < count and use_fallback:
            click.echo(f"\nGenerating fallback data...")
            fallback_texts = self.fallback.generate_texts(count * 2)
            for text in fallback_texts:
                records_out.append({
                    "url": "",
                    "text": text,
                    "source": "fallback",
                    "timestamp": datetime.now().isoformat()
                })
            click.echo(f"Generated {len(fallback_texts)} fallback documents")
        
        return records_out
    
    def save_texts(self, texts: List[Dict], filename: str = None) -> Path:
        """Save texts to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fetched_{timestamp}.jsonl"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in texts:
                text = record.get("text", "")
                if text and len(text.strip()) > 0:
                    json.dump({
                        "url": record.get("url", ""),
                        "text": text,
                        "source": record.get("source", ""),
                        "timestamp": record.get("timestamp", "")
                    }, f)
                    f.write("\n")
        
        return output_file


class DataFilter:
    """Filter data"""
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.seen_hashes: Set[str] = set()
        self.min_length = config.get("min_length", 200)
        self.max_length = config.get("max_length", 100000)
        self.min_words = config.get("min_words", 80)
        self.min_ascii_ratio = config.get("min_ascii_ratio", 0.98)
        self.min_english_stopwords = config.get("min_english_stopwords", 5)
        self.require_ascii = config.get("require_ascii", True)
        self.only_english = config.get("only_english", True)
        self.min_chars = config.get("min_chars", 1000)
    
    def filter_text(self, text: str, url: str = "") -> bool:
        if len(text) < self.min_chars:
            return False
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        if url and is_blacklisted_wiki_url(url):
            return False
        if not text:
            return False
        if self.require_ascii:
            ascii_chars = sum(1 for ch in text if ord(ch) < 128)
            ratio = ascii_chars / max(len(text), 1)
            if ratio < self.min_ascii_ratio:
                return False
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if len(words) < self.min_words:
            return False
        if self.only_english:
            stopword_hits = sum(1 for w in words if w.lower() in STOPWORDS)
            if stopword_hits < self.min_english_stopwords:
                return False
        h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
        if h in self.seen_hashes:
            return False
        self.seen_hashes.add(h)
        return True


class TextPreprocessor:
    """Preprocess text"""
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
    
    def clean_text(self, text: str) -> str:
        # Remove common Wikipedia sections to reduce noise
        text = re.sub(r"\n==\s*References\s*==.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\n==\s*External links\s*==.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\n==\s*See also\s*==.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\n==\s*Notes\s*==.*", "", text, flags=re.IGNORECASE | re.DOTALL)
        # Remove non-ASCII characters (foreign languages)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        # Only match English letters and basic punctuation
        tokens = re.findall(r'\b[a-z]+\b|[.,!?;:\'\"-]', text.lower())
        return tokens


class Vocabulary:
    """Vocabulary"""
    
    def __init__(self, max_vocab_size: int = 30000):
        self.max_vocab_size = max_vocab_size
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2token = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.token_counts = Counter()
    
    def build_from_texts(self, texts: List[str], preprocessor: TextPreprocessor):
        click.echo("Building vocabulary...")
        for text in tqdm(texts, desc="Processing"):
            tokens = preprocessor.tokenize(text)
            self.token_counts.update(tokens)
        
        most_common = self.token_counts.most_common(self.max_vocab_size - 4)
        for idx, (token, _) in enumerate(most_common, start=4):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        click.echo(f"Vocabulary size: {len(self.token2idx)}")
    
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(token, 1) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(idx, '<UNK>') for idx in indices]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token_counts': self.token_counts
            }, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.token2idx = data['token2idx']
            self.idx2token = data['idx2token']
            self.token_counts = data['token_counts']


class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, texts: List[str], vocab: Vocabulary, 
                 preprocessor: TextPreprocessor, seq_length: int = 16):
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.seq_length = seq_length
        self.sequences = []
        
        click.echo("Creating training sequences...")
        
        all_tokens = []
        for text in texts:
            tokens = preprocessor.tokenize(text)
            all_tokens.extend(tokens)
        
        click.echo(f"Total tokens: {len(all_tokens)}")
        
        all_indices = vocab.encode(all_tokens)
        
        for i in range(0, len(all_indices) - seq_length, 1):
            seq = all_indices[i:i + seq_length]
            if len(seq) == seq_length:
                self.sequences.append(seq)
        
        click.echo(f"Created {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class LSTMLanguageModel(nn.Module):
    """LSTM model"""
    
    def __init__(self, vocab_size: int, embed_size: int = 128, 
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden


class GRULanguageModel(nn.Module):
    """GRU model"""
    
    def __init__(self, vocab_size: int, embed_size: int = 128,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        gru_out, hidden = self.gru(embeds, hidden)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)
        return logits, hidden


class TransformerLM(nn.Module):
    """Transformer language model (BPE tokenized)"""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 4, ffn_dim: int = 512, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        bsz, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.embed(x) + self.pos(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask)
        logits = self.fc(h)
        return logits


def load_or_train_tokenizer(texts: List[str], tokenizer_path: Path, vocab_size: int = 8000) -> Tokenizer:
    if Tokenizer is None:
        raise RuntimeError("tokenizers is not installed. Run: pip install tokenizers")
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    # Train from in-memory texts by writing a temp file
    tmp_path = tokenizer_path.parent / "_tokenizer_corpus.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    tokenizer.train([str(tmp_path)], trainer)
    tmp_path.unlink(missing_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


class BpeTextDataset(Dataset):
    """Dataset for BPE tokenized language modeling"""

    def __init__(self, texts: List[str], tokenizer: Tokenizer, seq_length: int = 128):
        self.seq_length = seq_length
        self.sequences = []

        click.echo("Creating BPE training sequences...")
        ids = []
        for text in texts:
            ids.extend(tokenizer.encode(text).ids)

        click.echo(f"Total tokens: {len(ids)}")

        for i in range(0, len(ids) - seq_length - 1):
            seq = ids[i:i + seq_length + 1]
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)

        click.echo(f"Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class BpeTextGenerator:
    """Text generator for transformer BPE model"""

    def __init__(self, model, tokenizer: Tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8, top_k: int = 40):
        if not prompt:
            prompt = "<bos>"
        ids = self.tokenizer.encode(prompt).ids
        if not ids:
            ids = [self.tokenizer.token_to_id("<bos>") or 0]

        with torch.no_grad():
            for _ in range(max_length):
                context = ids[-256:]
                x = torch.tensor([context], dtype=torch.long).to(self.device)
                logits = self.model(x)
                logits = logits[:, -1, :] / max(temperature, 0.1)
                k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, k)
                probs = torch.softmax(top_k_logits, dim=-1)
                if not torch.isfinite(probs).all() or probs.sum().item() == 0:
                    # fallback to argmax to avoid NaNs
                    next_token = top_k_indices[0, 0].item()
                else:
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices[0, next_token_idx].item()
                ids.append(next_token)

        return self.tokenizer.decode(ids)


class Trainer:
    """Model trainer"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', use_amp: bool = False):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = bool(use_amp) and device.startswith("cuda")
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        click.echo(f"Using device: {device}")
    
    def train_epoch(self, dataloader, optimizer, epoch, grad_accum_steps: int = 1):
        self.model.train()
        total_loss = 0
        skipped = 0
        steps = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for step, (x, y) in enumerate(progress_bar, start=1):
            x, y = x.to(self.device), y.to(self.device)
            steps += 1
            try:
                autocast = torch.amp.autocast
                autocast_ctx = autocast('cuda', enabled=self.use_amp)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
            with autocast_ctx:
                out = self.model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / max(grad_accum_steps, 1)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad()
                continue
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if step % grad_accum_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * max(grad_accum_steps, 1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        if skipped == steps and steps > 0:
            return float('nan')
        return total_loss / max(len(dataloader), 1)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        skipped = 0
        steps = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                steps += 1
                try:
                    autocast = torch.amp.autocast
                    autocast_ctx = autocast('cuda', enabled=self.use_amp)
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)
                with autocast_ctx:
                    out = self.model(x)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                total_loss += loss.item()
        if skipped == steps and steps > 0:
            return float('nan')
        return total_loss / max(len(dataloader), 1)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
    
    def load_checkpoint(self, path: str, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']


class TextGenerator:
    """Text generator"""
    
    def __init__(self, model, vocab: Vocabulary, device='cpu'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 50, 
                temperature: float = 0.8, top_k: int = 40):
        preprocessor = TextPreprocessor(lowercase=True)
        tokens = preprocessor.tokenize(prompt)
        indices = self.vocab.encode(tokens)
        
        if not indices:
            indices = [self.vocab.token2idx.get('the', 1)]
        
        input_seq = torch.tensor([indices], dtype=torch.long).to(self.device)
        generated = indices.copy()
        
        with torch.no_grad():
            hidden = None
            for _ in range(max_length):
                logits, hidden = self.model(input_seq, hidden)
                logits = logits[:, -1, :] / max(temperature, 0.1)
                
                k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, next_token_idx].item()
                
                generated.append(next_token)
                input_seq = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                
                if next_token == self.vocab.token2idx.get('.', -1) and len(generated) > 20:
                    if random.random() < 0.3:
                        break
        
        decoded = self.vocab.decode(generated)
        text = ' '.join(decoded)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text


class AutoTrainer:
    """Automated trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.iteration = 0
        self.model_dir = Path(config.get('model_dir', 'models/auto'))
        self.data_dir = Path(config.get('data_dir', 'data/auto'))
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_manager = DataSourceManager(str(self.data_dir / 'raw'), config=self.config)
        
        self.default_stats = {
            'iterations': 0,
            'total_documents': 0,
            'total_sequences': 0,
            'total_training_time': 0,
            'best_loss': float('inf'),
        }
        self.stats = self.default_stats.copy()
        
        self.load_stats()
        self.replay_path = self.data_dir / 'replay.jsonl'
        self.eval_path = self.data_dir / 'val.jsonl'
        self.corpus_path = self.data_dir / 'corpus.txt'
        self.metrics_path = self.model_dir / 'metrics.csv'
    
    def load_stats(self):
        stats_file = self.model_dir / 'training_stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    loaded_stats = json.load(f)
                
                for key in self.default_stats:
                    if key in loaded_stats:
                        self.stats[key] = loaded_stats[key]
                    else:
                        self.stats[key] = self.default_stats[key]
                
                self.iteration = self.stats['iterations']
            except Exception as e:
                click.echo(f"Warning: Could not load stats: {e}")
    
    def save_stats(self):
        stats_file = self.model_dir / 'training_stats.json'
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            click.echo(f"Warning: Could not save stats: {e}")
    
    def fetch_data(self) -> Path:
        click.echo("\n" + "="*60)
        click.echo(f"ITERATION {self.iteration + 1}: FETCHING DATA")
        click.echo("="*60)
        
        count = self.config.get('docs_per_iteration', 50)
        texts = self.data_manager.fetch_data(count, use_fallback=True)
        output_file = self.data_manager.save_texts(texts)
        
        click.echo(f"\nTotal documents: {len(texts)}")
        return output_file
    
    def filter_data(self, input_file: Path) -> Path:
        click.echo("\n" + "="*60)
        click.echo(f"ITERATION {self.iteration + 1}: FILTERING DATA")
        click.echo("="*60)
        
        output_dir = self.data_dir / 'clean'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / input_file.name.replace('fetched', 'filtered')
        
        data_filter = DataFilter(self.config)
        preprocessor = TextPreprocessor()
        
        filtered_count = 0
        total_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                total_count += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    url = data.get('url', '')
                    
                    if data_filter.filter_text(text, url):
                        cleaned_text = preprocessor.clean_text(text)
                        if len(cleaned_text) > 200:
                            json.dump({'text': cleaned_text, 'url': url}, fout)
                            fout.write('\n')
                            filtered_count += 1
                            # Build replay buffer
                            self._append_replay(cleaned_text, url)
                            self._append_corpus(cleaned_text)
                except Exception:
                    pass
        
        self.stats['total_documents'] = self.stats.get('total_documents', 0) + filtered_count
        click.echo(f"Kept {filtered_count}/{total_count} documents")
        
        return output_file

    def _append_corpus(self, text: str):
        max_chars = self.config.get('corpus_max_chars', 5_000_000)
        if max_chars <= 0:
            return
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, 'a', encoding='utf-8') as f:
            f.write(text.replace("\n", " ").strip() + "\n")
        try:
            size = self.corpus_path.stat().st_size
            if size > max_chars:
                # Trim to last max_chars
                with open(self.corpus_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                data = data[-max_chars:]
                with open(self.corpus_path, 'w', encoding='utf-8') as f:
                    f.write(data)
        except Exception:
            pass

    def _load_corpus_sample(self, count: int) -> List[str]:
        if not self.corpus_path.exists() or count <= 0:
            return []
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines:
                return []
            sample = random.sample(lines, min(count, len(lines)))
            return [t.strip() for t in sample if t.strip()]
        except Exception:
            return []

    def _append_replay(self, text: str, url: str):
        max_docs = self.config.get('replay_max_docs', 2000)
        if max_docs <= 0:
            return
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.replay_path, 'a', encoding='utf-8') as f:
            json.dump({'text': text, 'url': url}, f)
            f.write('\n')
        # Trim if too large
        try:
            with open(self.replay_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > max_docs:
                lines = lines[-max_docs:]
                with open(self.replay_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
        except Exception:
            pass

    def _load_replay(self, count: int) -> List[str]:
        if not self.replay_path.exists() or count <= 0:
            return []
        try:
            with open(self.replay_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if not lines:
                return []
            sample = random.sample(lines, min(count, len(lines)))
            texts = []
            for line in sample:
                try:
                    data = json.loads(line)
                    texts.append(data.get('text', ''))
                except Exception:
                    continue
            return [t for t in texts if t]
        except Exception:
            return []

    def _ensure_eval_set(self, texts: List[str]):
        if self.eval_path.exists() or not texts:
            return
        eval_count = min(self.config.get('eval_set_size', 50), len(texts))
        sample = random.sample(texts, eval_count)
        self.eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.eval_path, 'w', encoding='utf-8') as f:
            for t in sample:
                json.dump({'text': t}, f)
                f.write('\n')

    def _load_eval_texts(self) -> List[str]:
        if not self.eval_path.exists():
            return []
        texts = []
        with open(self.eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    texts.append(data.get('text', ''))
                except Exception:
                    continue
        return [t for t in texts if t]
    
    def train_model(self, data_file: Path):
        click.echo("\n" + "="*60)
        click.echo(f"ITERATION {self.iteration + 1}: TRAINING MODEL")
        click.echo("="*60)
        
        start_time = time.time()
        
        texts = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                except Exception:
                    pass
        
        if len(texts) < 1:
            click.echo("No data to train.")
            return
        
        click.echo(f"Loaded {len(texts)} documents")

        # Build replay + fixed eval set
        self._ensure_eval_set(texts)
        replay_ratio = self.config.get('replay_ratio', 0.3)
        if replay_ratio > 0:
            replay_count = int(len(texts) * replay_ratio)
            replay_texts = self._load_replay(replay_count)
            if replay_texts:
                texts.extend(replay_texts)
                click.echo(f"Added {len(replay_texts)} replay documents")
        corpus_ratio = self.config.get('corpus_ratio', 0.3)
        if corpus_ratio > 0:
            corpus_count = int(len(texts) * corpus_ratio)
            corpus_texts = self._load_corpus_sample(corpus_count)
            if corpus_texts:
                texts.extend(corpus_texts)
                click.echo(f"Added {len(corpus_texts)} corpus documents")

        # Dialogue formatting
        if self.config.get('dialogue_format', False):
            prompts = self.config.get('dialogue_prompts', [
                "Hello!",
                "What is this about?",
                "Explain it briefly.",
                "Can you summarize?",
                "Tell me more.",
            ])
            multi_turn = self.config.get('dialogue_multiturn', True)
            if multi_turn:
                texts = [
                    f"User: {random.choice(prompts)}\nAssistant: {t}\n"
                    f"User: {random.choice(prompts)}\nAssistant: {t} <END>"
                    for t in texts
                ]
            else:
                texts = [f"User: {random.choice(prompts)}\nAssistant: {t} <END>" for t in texts]
        
        vocab_path = self.model_dir / 'vocab.pkl'
        tokenizer_path = self.model_dir / 'tokenizer.json'
        preprocessor = TextPreprocessor(lowercase=True)
        model_type = self.config.get('model_type', 'lstm')
        resume = self.config.get('resume', True)

        seq_length = self.config.get('seq_length', 16)
        if model_type == 'transformer_bpe':
            tokenizer = load_or_train_tokenizer(
                texts,
                tokenizer_path,
                vocab_size=self.config.get('vocab_size', 8000)
            )
            dataset = BpeTextDataset(texts, tokenizer, seq_length=seq_length)
        else:
            vocab = Vocabulary(max_vocab_size=self.config.get('vocab_size', 30000))
            if resume and vocab_path.exists():
                vocab.load(vocab_path)
            else:
                vocab.build_from_texts(texts, preprocessor)
                vocab.save(vocab_path)
            dataset = TextDataset(texts, vocab, preprocessor, seq_length=seq_length)
        
        if len(dataset) < 1:
            click.echo("No sequences created.")
            return
        
        self.stats['total_sequences'] = self.stats.get('total_sequences', 0) + len(dataset)
        
        train_size = max(1, int(0.9 * len(dataset)))
        val_size = len(dataset) - train_size
        
        if val_size < 1:
            train_dataset = dataset
            val_dataset = dataset
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        
        # Device selection (needed before DataLoader pin_memory)
        device_pref = self.config.get('device', 'auto')
        if device_pref == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_pref

        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 0)
        if num_workers < 0:
            try:
                cpu_count = os.cpu_count() or 2
                num_workers = max(2, min(8, cpu_count - 2))
            except Exception:
                num_workers = 2
        pin_memory = device.startswith('cuda')
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        embed_size = self.config.get('embed_size', 128)
        hidden_size = self.config.get('hidden_size', 256)
        num_layers = self.config.get('num_layers', 2)

        if model_type == 'transformer_bpe':
            model = TransformerLM(
                vocab_size=tokenizer.get_vocab_size(),
                d_model=self.config.get('d_model', 256),
                nhead=self.config.get('nhead', 4),
                num_layers=self.config.get('num_layers', 4),
                ffn_dim=self.config.get('ffn_dim', 512),
                dropout=self.config.get('dropout', 0.1),
                max_len=max(seq_length, 16),
            )
        elif model_type == 'lstm':
            model = LSTMLanguageModel(len(vocab.token2idx), embed_size, hidden_size, num_layers)
        else:
            model = GRULanguageModel(len(vocab.token2idx), embed_size, hidden_size, num_layers)
        
        trainer = Trainer(model, device=device, use_amp=self.config.get('use_amp', True))
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        weight_decay = self.config.get('weight_decay', 0.01)
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('lr', 0.001))
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.get('lr', 0.001), weight_decay=weight_decay)
        
        epochs = self.config.get('epochs_per_iteration', 5)
        best_val_loss = float(self.stats.get('best_loss', float('inf')))
        checkpoint_path = self.model_dir / 'best_model.pt'
        latest_path = self.model_dir / 'latest_model.pt'
        patience = self.config.get('early_stop_patience', 3)
        bad_epochs = 0

        if resume and latest_path.exists():
            try:
                checkpoint = torch.load(str(latest_path), map_location=trainer.device)
                state = checkpoint.get('model_state_dict', checkpoint)
                missing, unexpected = model.load_state_dict(state, strict=False)
                if optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                click.echo("Resumed from latest checkpoint (strict=False).")
            except Exception:
                pass
        
        # Create generator for epoch samples
        if model_type == 'transformer_bpe':
            generator = BpeTextGenerator(model, tokenizer, device=trainer.device)
        else:
            generator = TextGenerator(model, vocab, device=trainer.device)
        prompts = ["the study of", "scientists have", "research shows", "it is known"]
        sample_len_epoch = self.config.get('sample_len_epoch', 40)
        
        eval_texts = self._load_eval_texts()
        if model_type == 'transformer_bpe':
            eval_dataset = BpeTextDataset(eval_texts, tokenizer, seq_length=seq_length) if eval_texts else None
        else:
            eval_dataset = TextDataset(eval_texts, vocab, preprocessor, seq_length=seq_length) if eval_texts else None
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory) if eval_dataset and len(eval_dataset) > 0 else None

        grad_accum_steps = self.config.get('grad_accum_steps', 1)
        for epoch in range(1, epochs + 1):
            train_loss = trainer.train_epoch(train_loader, optimizer, epoch, grad_accum_steps=grad_accum_steps)
            val_loss = trainer.evaluate(val_loader)
            eval_loss = trainer.evaluate(eval_loader) if eval_loader else None
            
            try:
                ppl = math.exp(val_loss) if math.isfinite(val_loss) else float('inf')
            except Exception:
                ppl = float('inf')
            if eval_loss is not None:
                click.echo(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Eval={eval_loss:.4f}, PPL={ppl:.2f}")
            else:
                click.echo(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, PPL={ppl:.2f}")

            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                click.echo("Non-finite loss detected; skipping sample generation this epoch.")
                break

            # Metrics log
            try:
                if not self.metrics_path.exists():
                    with open(self.metrics_path, 'w', encoding='utf-8') as f:
                        f.write("iteration,epoch,train_loss,val_loss,eval_loss,ppl\n")
                with open(self.metrics_path, 'a', encoding='utf-8') as f:
                    f.write(f"{self.iteration},{epoch},{train_loss:.6f},{val_loss:.6f},{(eval_loss if eval_loss is not None else '')},{ppl:.6f}\n")
            except Exception:
                pass
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.stats['best_loss'] = float(best_val_loss)
                trainer.save_checkpoint(checkpoint_path, epoch, optimizer, val_loss)
                click.echo(f"New best model! Loss: {val_loss:.4f}")
                bad_epochs = 0
            else:
                bad_epochs += 1

            trainer.save_checkpoint(latest_path, epoch, optimizer, val_loss)
            
            # Generate sample after each epoch
            prompt = random.choice(prompts)
            sample = generator.generate(prompt, max_length=sample_len_epoch, temperature=0.7)
            click.echo(f"Sample: {sample}")
            click.echo("-" * 40)

            if patience and bad_epochs >= patience:
                click.echo(f"Early stopping after {bad_epochs} bad epochs")
                break
        
        training_time = time.time() - start_time
        self.stats['total_training_time'] = self.stats.get('total_training_time', 0) + training_time
        click.echo(f"Training complete in {training_time:.2f}s")
    
    def generate_sample(self):
        click.echo("\n" + "="*60)
        click.echo(f"ITERATION {self.iteration + 1}: SAMPLE GENERATION")
        click.echo("="*60)
        
        try:
            vocab_path = self.model_dir / 'vocab.pkl'
            checkpoint_path = self.model_dir / 'best_model.pt'
            tokenizer_path = self.model_dir / 'tokenizer.json'
            
            if not checkpoint_path.exists():
                click.echo("Model not ready.")
                return
            
            model_type = self.config.get('model_type', 'lstm')
            embed_size = self.config.get('embed_size', 128)
            hidden_size = self.config.get('hidden_size', 256)
            num_layers = self.config.get('num_layers', 2)
            
            device_pref = self.config.get('device', 'auto')
            if device_pref == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = device_pref

            if model_type == 'transformer_bpe':
                if not tokenizer_path.exists():
                    click.echo("Tokenizer not found.")
                    return
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                model = TransformerLM(
                    vocab_size=tokenizer.get_vocab_size(),
                    d_model=self.config.get('d_model', 256),
                    nhead=self.config.get('nhead', 4),
                    num_layers=self.config.get('num_layers', 4),
                    ffn_dim=self.config.get('ffn_dim', 512),
                    dropout=self.config.get('dropout', 0.1),
                    max_len=max(self.config.get('seq_length', 128), 16),
                )
                generator = BpeTextGenerator(model, tokenizer, device=device)
            else:
                if not vocab_path.exists():
                    click.echo("Vocab not found.")
                    return
                vocab = Vocabulary()
                vocab.load(vocab_path)
                if model_type == 'lstm':
                    model = LSTMLanguageModel(len(vocab.token2idx), embed_size, hidden_size, num_layers)
                else:
                    model = GRULanguageModel(len(vocab.token2idx), embed_size, hidden_size, num_layers)
                generator = TextGenerator(model, vocab, device=device)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state, strict=False)
            
            prompts = ["the study of", "scientists have", "research shows", "it is known"]
            prompt = random.choice(prompts)
            
            sample_len_final = self.config.get('sample_len_final', 120)
            generated = generator.generate(prompt, max_length=sample_len_final, temperature=0.7)
            
            click.echo(f"\nPrompt: '{prompt}'")
            click.echo(f"Generated: {generated}\n")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    def run_iteration(self):
        self.iteration += 1
        self.stats['iterations'] = self.iteration
        
        click.echo("\n" + "="*60)
        click.echo(f"STARTING ITERATION {self.iteration}")
        click.echo("="*60)
        
        try:
            raw_data = self.fetch_data()
            clean_data = self.filter_data(raw_data)
            self.train_model(clean_data)
            self.generate_sample()
            self.save_stats()
            
            click.echo("\n" + "="*60)
            click.echo("SUMMARY")
            click.echo("="*60)
            click.echo(f"  Iterations: {self.stats.get('iterations', 0)}")
            click.echo(f"  Documents: {self.stats.get('total_documents', 0)}")
            click.echo(f"  Sequences: {self.stats.get('total_sequences', 0)}")
            best_loss = self.stats.get('best_loss', float('inf'))
            if best_loss != float('inf'):
                click.echo(f"  Best loss: {best_loss:.4f}")
            click.echo("="*60)
        except KeyboardInterrupt:
            self.save_stats()
            raise
        except Exception as e:
            click.echo(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            self.save_stats()
    
    def run_loop(self, max_iterations: int = None):
        click.echo("\nAUTOMATED TRAINING MODE\n")
        
        try:
            iteration = 0
            while max_iterations is None or iteration < max_iterations:
                self.run_iteration()
                
                wait_time = self.config.get('wait_between_iterations', 0)
                if wait_time and wait_time > 0:
                    click.echo(f"\nWaiting {wait_time}s...\n")
                    time.sleep(wait_time)
                
                iteration += 1
        except KeyboardInterrupt:
            click.echo("\n\nStopped")
            self.save_stats()


def get_auto_config() -> dict:
    config_file = Path('auto_config.json')
    
    default_config = {
        'model_dir': 'models/auto',
        'data_dir': 'data/auto',
        'docs_per_iteration': 50,
        'model_type': 'transformer_bpe',
        'resume': True,
        'device': 'auto',
        'embed_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'd_model': 256,
        'nhead': 4,
        'ffn_dim': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'num_workers': 2,
        'epochs_per_iteration': 5,
        'lr': 0.001,
        'seq_length': 256,
        'vocab_size': 16000,
        'use_amp': True,
        'early_stop_patience': 3,
        'replay_ratio': 0.3,
        'replay_max_docs': 2000,
        'eval_set_size': 50,
        'corpus_ratio': 0.3,
        'corpus_max_chars': 5_000_000,
        'optimizer': 'adamw',
        'weight_decay': 0.01,
        'grad_accum_steps': 1,
        'sample_len_epoch': 40,
        'sample_len_final': 120,
        'wait_between_iterations': 0,
        'use_mediawiki': True,
        'mediawiki_api': 'https://en.wikipedia.org/w/api.php',
        'mediawiki_endpoints': [
            'https://en.wikipedia.org/w/api.php',
            'https://simple.wikipedia.org/w/api.php',
            'https://en.wikibooks.org/w/api.php',
            'https://en.wikinews.org/w/api.php',
            'https://en.wiktionary.org/w/api.php',
            'https://en.wikiquote.org/w/api.php',
            'https://en.wikisource.org/w/api.php',
            'https://en.wikiversity.org/w/api.php',
            'https://en.wikivoyage.org/w/api.php',
        ],
        'use_stackexchange': True,
        'stackexchange_site': 'stackoverflow',
        'stackexchange_docs': 50,
        'stackexchange_key': '',
        'stackexchange_qa': True,
        'use_commoncrawl': False,
        'dialogue_format': True,
        'dialogue_multiturn': True,
        'dialogue_prompts': [
            'Hello!',
            'What is this about?',
            'Can you explain?',
            'Summarize this.',
            'Tell me more.'
        ],
        'mediawiki_categories': [
            'Science',
            'Technology',
            'Mathematics',
            'Physics',
            'Biology',
            'Chemistry',
            'History'
        ],
        'use_pageviews': True,
        'pageviews_days_back': 1,
        'min_length': 200,
        'max_length': 100000,
        'min_words': 80,
        'min_ascii_ratio': 0.98,
        'min_english_stopwords': 5,
        'require_ascii': True,
        'only_english': True,
        'url_pattern': '',
        'match_type': 'domain',
        'cc_index': '',
        'cc_filter': 'status:200',
        'domains': ['wikipedia.org'],
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                for key, value in default_config.items():
                    if key not in loaded:
                        loaded[key] = value
                return loaded
        except Exception:
            pass
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    return default_config


def run_chat(config: dict):
    model_dir = Path(config.get('model_dir', 'models/auto'))
    vocab_path = model_dir / 'vocab.pkl'
    tokenizer_path = model_dir / 'tokenizer.json'
    checkpoint_path = model_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / 'latest_model.pt'

    if not checkpoint_path.exists():
        click.echo("No model checkpoint found.")
        return

    device = config.get('device', 'cpu')
    model_type = config.get('model_type', 'lstm')
    max_len = config.get('sample_len_final', 120)

    if model_type == 'transformer_bpe':
        if not tokenizer_path.exists():
            click.echo("Tokenizer not found.")
            return
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        model = TransformerLM(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 4),
            ffn_dim=config.get('ffn_dim', 512),
            dropout=config.get('dropout', 0.1),
            max_len=max(config.get('seq_length', 128), 16),
        )
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
        generator = BpeTextGenerator(model, tokenizer, device=device)
    else:
        if not vocab_path.exists():
            click.echo("Vocab not found.")
            return
        vocab = Vocabulary()
        vocab.load(vocab_path)
        if model_type == 'lstm':
            model = LSTMLanguageModel(len(vocab.token2idx), config.get('embed_size', 128),
                                      config.get('hidden_size', 256), config.get('num_layers', 2))
        else:
            model = GRULanguageModel(len(vocab.token2idx), config.get('embed_size', 128),
                                     config.get('hidden_size', 256), config.get('num_layers', 2))
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
        generator = TextGenerator(model, vocab, device=device)

    click.echo("\nChat mode. Type 'exit' to quit.\n")
    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        if config.get('dialogue_format', False):
            prompt_in = f"User: {prompt}\nAssistant:"
        else:
            prompt_in = prompt
        reply = generator.generate(prompt_in, max_length=max_len, temperature=0.7)
        if config.get('dialogue_format', False):
            # Strip any echoed prefix
            reply = reply.replace(prompt_in, "").strip()
            if "<END>" in reply:
                reply = reply.split("<END>", 1)[0].strip()
        click.echo(f"AI> {reply}\n")


@click.command()
@click.option('--iterations', default=0, help='Number of iterations (0=infinite)')
@click.option('--test', is_flag=True, help='Test Common Crawl')
@click.option('--gpu', is_flag=True, help='Use GPU (CUDA) if available')
@click.option('--chat', is_flag=True, help='Interactive chat with the model')
def main(iterations, test, gpu, chat):
    """LLM Trainer"""
    
    click.echo("="*60)
    click.echo("LLM TRAINING TOOL")
    click.echo("="*60 + "\n")
    
    if test:
        click.echo("Testing Common Crawl...\n")
        fetcher = ImprovedCommonCrawlFetcher()
        
        try:
            count = 0
            for record in fetcher.fetch_from_index("wikipedia.org", max_records=3):
                count += 1
                click.echo(f"\nRecord {count}:")
                click.echo(f"  URL: {record['url'][:60]}...")
                click.echo(f"  Length: {len(record['text'])} chars")
                click.echo(f"  Preview: {record['text'][:200]}...")
            
            if count > 0:
                click.echo(f"\nFetched {count} records!")
            else:
                click.echo("\nNo records fetched")
        except Exception as e:
            click.echo(f"\nError: {e}")
        
        return
    
    config = get_auto_config()
    # CPU by default; override with --gpu
    if gpu:
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'

    if chat:
        run_chat(config)
        return

    trainer = AutoTrainer(config)
    trainer.run_loop(None if iterations == 0 else iterations)


if __name__ == '__main__':
    main()
