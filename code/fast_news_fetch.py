import os
import csv
import time
import requests
import json
import hashlib
from datetime import datetime, timedelta
from dateutil.parser import parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configuration with optimized symbol options
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
SYMBOL_OPTIONS = {
    "EURUSD": ["EURUSD.FOREX"],
    "GBPUSD": ["GBPUSD.FOREX"],
    "USDJPY": ["JPY.FOREX", "USDJPY.FOREX"],
    "AUDUSD": ["AUDUSD.FOREX"],
    "USDCAD": ["CADUSD.FOREX", "CAD.FOREX"],
    "XAUUSD": ["GOLD", "XAU.FOREX"]
}
DATA_DIR = r"C:\FPFX\data"
OUTPUT_CSV = os.path.join(DATA_DIR, "news_cache.csv")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "news_fetch_checkpoint.json")
API_KEY = "684d76d81e83b0.88582643"
API_URL = "https://eodhd.com/api/news"
RATE_LIMIT_S = 1.5  # Conservative delay
BATCH_SIZE = 500     # Optimal for reliability

# Initialize
os.makedirs(DATA_DIR, exist_ok=True)
sia = SentimentIntensityAnalyzer()
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=730)  # 2 years

# Load checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"completed_symbols": [], "symbol_options": {}, "article_counts": {}}

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f)

# Create article hash for deduplication
def get_article_hash(article):
    title = article.get('title', '')[:100]  # First 100 chars
    date = article.get('date', '')[:10]     # YYYY-MM-DD only
    return hashlib.md5(f"{title}|{date}".encode()).hexdigest()

# Main fetching function - NOW FETCHES ALL AVAILABLE ARTICLES
def fetch_articles(symbol):
    checkpoint = load_checkpoint()
    articles = []
    seen_hashes = set()
    
    # Skip if already completed
    if symbol in checkpoint["completed_symbols"]:
        print(f"  {symbol} already completed - skipping")
        return checkpoint["article_counts"].get(symbol, [])
    
    # Use best-known symbol option or default list
    options = checkpoint["symbol_options"].get(symbol, SYMBOL_OPTIONS[symbol].copy())
    best_symbol = None
    
    for symbol_code in options:
        print(f"  Trying symbol: {symbol_code}")
        offset = 0
        page = 1
        symbol_success = True
        page_errors = 0  # Track consecutive errors per page
        
        while True:
            params = {
                "api_token": API_KEY,
                "s": symbol_code,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "offset": offset,
                "limit": BATCH_SIZE,
                "fmt": "json"
            }
            
            try:
                resp = requests.get(API_URL, params=params, timeout=60)
                if resp.status_code != 200:
                    print(f"    Status {resp.status_code} - skipping page {page} (attempt {page_errors + 1}/3)")
                    
                    # Skip page after 3 failures
                    if page_errors >= 2:
                        print(f"    Permanent error on page {page} - skipping to next page")
                        offset += BATCH_SIZE
                        page += 1
                        page_errors = 0
                        time.sleep(5)
                        continue
                    
                    page_errors += 1
                    time.sleep(5)  # Longer wait on errors
                    continue
                    
                # Reset error counter on success
                page_errors = 0
                new_articles = resp.json()
                if not new_articles:
                    print(f"    No more articles at page {page}")
                    break
                
                # Filter future articles
                current_date = datetime.utcnow().date()
                valid_articles = []
                for art in new_articles:
                    try:
                        art_date = parse(art['date']).date()
                        if art_date <= current_date:
                            valid_articles.append(art)
                    except:
                        continue
                
                # Deduplicate
                new_count = 0
                for art in valid_articles:
                    art_hash = get_article_hash(art)
                    if art_hash not in seen_hashes:
                        seen_hashes.add(art_hash)
                        articles.append(art)
                        new_count += 1
                
                print(f"    Page {page}: Added {new_count} articles (total: {len(articles)})")
                
                # Always continue to next page if there might be more data
                if len(new_articles) < BATCH_SIZE:
                    print("    Last page detected")
                    break
                    
                offset += len(new_articles)
                page += 1
                time.sleep(RATE_LIMIT_S)
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                symbol_success = False
                break
        
        # If successful with this symbol, remember it for future
        if symbol_success and articles:
            best_symbol = symbol_code
            break
    
    # Update checkpoint
    if best_symbol:
        checkpoint["symbol_options"][symbol] = [best_symbol]
    checkpoint["completed_symbols"].append(symbol)
    checkpoint["article_counts"][symbol] = articles
    save_checkpoint(checkpoint)
    
    return articles

# Main Execution
print("Starting comprehensive news fetch...")
print(f"Gathering ALL available articles for {len(SYMBOLS)} symbols")
checkpoint = load_checkpoint()
all_articles = {}

for symbol in SYMBOLS:
    print(f"\n{'='*40}")
    print(f"Fetching {symbol} news...")
    
    articles = fetch_articles(symbol)
    all_articles[symbol] = articles
    print(f"Total {symbol} articles: {len(articles)}")

# Sentiment processing and CSV output
print("\nProcessing sentiment scores...")
date_records = {}

for symbol, articles in all_articles.items():
    for article in articles:
        try:
            pub_date = parse(article['date']).date()
            date_str = pub_date.strftime("%Y-%m-%d")
            title = article.get('title', '')
            content = article.get('content', '') or article.get('description', '')
            text = f"{title}. {content}"[:2000]  # Truncate long content
            
            # Get sentiment
            vs = sia.polarity_scores(text)
            sentiment = vs['compound']
            
            if date_str not in date_records:
                date_records[date_str] = {s: [] for s in SYMBOLS}
            date_records[date_str][symbol].append(sentiment)
        except Exception as e:
            continue

print("\nGenerating output...")
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "symbol", "news_count", "avg_sentiment"])
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        for symbol in SYMBOLS:
            sentiments = date_records.get(date_str, {}).get(symbol, [])
            count = len(sentiments)
            avg_sent = sum(sentiments)/count if count else 0.0
            writer.writerow([date_str, symbol, count, f"{avg_sent:.6f}"])
        current_date += timedelta(days=1)

print(f"✅ Success! Data saved to {OUTPUT_CSV}")

# Final summary
total_articles = sum(len(articles) for articles in all_articles.values())
print(f"\nFetched {total_articles} total articles")
for symbol, articles in all_articles.items():
    if articles:
        first_date = parse(articles[0]['date']).strftime('%Y-%m-%d')
        last_date = parse(articles[-1]['date']).strftime('%Y-%m-%d')
        print(f"  {symbol}: {len(articles)} articles ({first_date} to {last_date})")
    else:
        print(f"  {symbol}: 0 articles (no data found)")