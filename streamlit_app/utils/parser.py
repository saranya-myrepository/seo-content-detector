from __future__ import annotations
import argparse, csv, random, time, os
from typing import Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------- Config ----------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT_SEC = 12
BASE_DELAY = 1.2        # base delay; jitter added to land ~1.2–1.8s
RETRIES = 2             # retry attempts on failure

# ---------- Input loader (CSV or Excel) ----------
# (NO CHANGES TO load_urls_table - it is robust)
def load_urls_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    df = None
    # try CSV first
    if path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=None)
        except Exception:
            df = None
    # try Excel
    if (df is None or df.empty) and path.lower().endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(path)  # requires openpyxl for .xlsx
        except Exception:
            df = None
    # headerless CSV fallback (first column = url)
    if df is None or df.empty:
        try:
            df = pd.read_csv(path, header=None, names=["url"], encoding="utf-8-sig", engine="python", sep=None)
        except Exception:
            df = None
    # raw lines fallback
    if df is None or df.empty:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = [ln.strip() for ln in f]
        df = pd.DataFrame({"url": [ln for ln in lines if ln.startswith(("http://", "https://"))]})

    if df is None or df.empty:
        raise ValueError("Could not parse any URLs. Ensure a 'url' column or one http(s) URL per line.")

    # normalize to a single 'url' column
    if "url" not in df.columns:
        for c in df.columns:
            if str(c).strip().lower() == "url":
                df = df.rename(columns={c: "url"})
                break
        else:
            df = df.rename(columns={df.columns[0]: "url"})

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.startswith(("http://", "https://"))].drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError("After cleaning, no valid http(s) URLs were found.")
    return df

# ---------- HTML parsing (NO CHANGES) ----------
def _strip_layout_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        tag.decompose()

def parse_html(html: str) -> Tuple[Optional[str], str]:
    # NOTE: Since lxml is in your requirements, this is a good choice for speed.
    soup = BeautifulSoup(html, "lxml") 
    _strip_layout_noise(soup)
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    main = soup.find("main") or soup.find("article")
    if main:
        parts = [t.get_text(" ", strip=True) for t in main.find_all(["p","h1","h2","h3","li"])]
    else:
        parts = [t.get_text(" ", strip=True) for t in soup.find_all("p")]
        if not parts:
            parts = [soup.get_text(" ", strip=True)]
    body = " ".join(" ".join(parts).split())
    return title, body

# ---------- polite fetch (MINOR CHANGE) ----------
def fetch_html(url: str, session: requests.Session) -> Tuple[Optional[str], Optional[int]]:
    delay = BASE_DELAY + random.uniform(0.2, 0.6)  # ~1.4–1.8s typical
    for attempt in range(RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
            
            # Basic error handling for timeouts/404s (try-except blocks) is partially here
            if resp.status_code == 200 and resp.text:
                time.sleep(delay)  # polite delay after each successful request
                return resp.text, resp.status_code
            
            # Log specific HTTP errors (4xx or 5xx)
            if resp.status_code >= 400:
                print(f"  [HTTP ERROR] {resp.status_code} for {url} on attempt {attempt+1}")
                if resp.status_code < 500: # Don't retry client errors (4xx)
                    return None, resp.status_code

        except requests.RequestException as e:
            # Handle connection errors or timeouts
            print(f"  [CONN ERROR] {e.__class__.__name__} for {url} on attempt {attempt+1}")

        if attempt < RETRIES:
            time.sleep(delay * (1.5 ** attempt))  # simple backoff
    
    # If all retries fail
    final_status = resp.status_code if 'resp' in locals() else None
    return None, final_status

# ---------- main (UPDATED LOGGING) ----------
def scrape_urls(input_path: str, output_csv: str) -> None:
    df = load_urls_table(input_path)
    urls = df["url"].tolist()
    rows = []

    print(f"Starting scrape of {len(urls)} URLs from {input_path}")
    
    with requests.Session() as sess:
        # Use simple enumeration instead of tqdm if it's causing display issues
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Fetching: {url}")
            
            # Fetch HTML content. Returns (html_string, status_code)
            html, status = fetch_html(url, sess)
            
            row_data = {"url": url, "title": None, "body_text": "", "word_count": 0}

            if html:
                try:
                    # Parse HTML content directly using appropriate Python libraries 
                    title, body = parse_html(html)
                    row_data["title"] = title
                    row_data["body_text"] = body
                    # Calculate word count from extracted text 
                    row_data["word_count"] = len(body.split())
                    print(f"  [SUCCESS] Words: {row_data['word_count']}")
                except Exception as e:
                    # Handle parsing errors gracefully (try-except blocks) [cite: 59]
                    print(f"  [PARSE FAIL] Error for {url}: {e.__class__.__name__}")
                    row_data["title"] = f"PARSE ERROR"
            else:
                row_data["title"] = f"FETCH FAILED (Status: {status})"
                print(f"  [FAILURE] Could not fetch content. Status: {status}")

            rows.append(row_data)

    out = pd.DataFrame(rows, columns=["url","title","body_text","word_count"])
    out_valid = out[out['word_count'] > 0] # Filter out complete failures for final count
    
    # Save extracted data to CSV format [cite: 62]
    out.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL) 
    print(f"\n--- Scraping Complete ---")
    print(f"Successfully processed {len(out)} entries.")
    print(f"Saved {len(out_valid)} valid content entries to: {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape URLs (CSV/Excel) → title/body/word_count CSV.")
    ap.add_argument("input_path", help="CSV or Excel file with 'url' column (e.g., data/urls.xlsx)")
    ap.add_argument("output_csv", help="Where to save extracted_content.csv")
    args = ap.parse_args()
    scrape_urls(args.input_path, args.output_csv)