import requests
import urllib.robotparser
import urllib
import json
import re
import hashlib
import torch
import os
import httpx
import numpy as np
import h5py

from bs4 import BeautifulSoup
from requests_html import HTMLSession
from boilerpy3 import extractors
from transformers import BertTokenizer, BertModel
from collections import defaultdict, Counter
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

def is_crawlable(url):
    try:
        parsed_uri = urllib.parse.urlparse(url)
        host_url = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        robot_parser = urllib.robotparser.RobotFileParser()
        robot_parser.set_url(host_url)
        robot_parser.read()
        crawlable = robot_parser.can_fetch("*", url)
    except Exception as e:
        print(e)
        return False 
    # return True
    return crawlable

def is_valid_mime_type(url):
    invalid_types = [".pdf", ".mp4", ".mp3", ".png", ".jpg", ".svg", ".xlsx", ".doc",
                        ".gif", ".jpeg", ".sh", ".tar", ".zip", ".ashx", ".tif", ".gz",
                        ".pptx"]
    for type in invalid_types:
        if type in url:
            return False
    return True

def fetch_search_engine_seeds(query):
    links = []
    links.extend(fetch_bing_links(query))
    links.extend(fetch_google_links(query))
    # links.extend(fetch_ecosia_links(query))
    return links

def fetch_google_links(query):
    # has to be valid header and cookie
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "cookie": "SID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnV_He2m6ZCg7iR02oHDRqPewACgYKAYkSAQASFQHGX2MiJYIvsk_W3TwQd6hkGyoXkRoVAUF8yKrxv4be_c6KgoeoPL7LgenE0076; __Secure-1PSID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnVeC-P0B2SDEl9SwDuQ1oMSwACgYKAT4SAQASFQHGX2MidKMmb45almtnnSZjHZj8qBoVAUF8yKqrsDEtenJokob51hABeKGk0076; __Secure-3PSID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnVZu6DHG5QsisGTUGQJ95-mQACgYKAb0SAQASFQHGX2MihLJVP806UgrORtDNhbvMIRoVAUF8yKpCr3qW5WKaoNADtEKMV9-g0076; HSID=A9quHzDj7YtkvFXZf; SSID=AfA-_ClZswKnBQu5a; APISID=fCwuPizlA06PC3Lh/AAIv18dK4y-njgHY7; SAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; __Secure-1PAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; __Secure-3PAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; NID=512=SXob5RvKanDNXYpfZ_lJIwbPcA-Ri2GBFeEpNY6kpC1rNsp-RY9PW9CQbffzJW1hbtK-v2PhOwIg61B9sveljNBq6ssETXkEaCvnO0gZDLLP9tOnzfpinnj6aqRIwPg2SydQ2k046pDv0OeLo3WHVVlY3Fh2Q88Zd8IpL-3qDtwh338uhr4P4fuv0xtZI-9t9_qSwEgYAStPTKBHDwXQNkPy1SfC3JacWalkbPNldmut5GbSeo7D__oDv1M8TQN-Budxvad1hvs-OlqnHw; AEC=AQTF6Hy7nE_iLLiGVNZyPoQ29N26OCVAeMStfphG6v5SPZLAETw_nvmcMg; __Secure-ENID=19.SE=QjOlIRWLXy-fJswsAaHbBf0WY_ZW-wonXTvx5HkLluiUIh91N1PyI_TOV-a5BGO5y3HqymYFTEhcqLmKNwxGsfSBf9CSGrZSGEfcPUPnX0YzPgqJIE9gjEswAG8ObjhIse6tBowKOMwVedcjvaOaZp0dRyoVyEamnI9liHSyIbynUc-9ygHS_pqaVOE180nyhC6Y7AMeashAKwh5J1XK7KSIEnZ7Q5XxMdQ1pj-GZxty8-7W-IvTHkCVUB48YmVwkwze0zcImdle_G2MPsyvn8c; OGPC=19041161-1:19008535-1:19026797-2:; OGP=-19008535:-19026797:",
        "dnt": "1",
        "referer": "https://www.google.de/",
        "sec-ch-ua": '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        "sec-ch-ua-arch": '"x86"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-full-version": '"123.0.2420.97"',
        "sec-ch-ua-full-version-list": '"Microsoft Edge";v="123.0.2420.97", "Not:A-Brand";v="8.0.0.0", "Chromium";v="123.0.6312.123"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": '""',
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua-platform-version": '"10.0.0"',
        "sec-ch-ua-wow64": "?0",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
    }

    params = {
        'q': query,  
        'hl': 'en',             
        'gl': 'us',             
        'num': '500'             
    }

    try:
        session = HTMLSession()
        response = session.get("https://www.google.com/search", headers=headers, params=params)
    except Exception as e:
        print(e)
        return None
    
    initial_urls = list(response.html.absolute_links)#[:25]
    filtered_urls = [x for x in initial_urls if "google" not in x]
    seed_urls = [x for x in filtered_urls if is_crawlable(x) and is_valid_mime_type(x)]
    return seed_urls

def fetch_bing_links(query):
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "avail-dictionary": "kJo4nTps",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
        "sec-ch-ua-arch": "\"x86\"",
        "sec-ch-ua-bitness": "\"64\"",
        "sec-ch-ua-full-version": "\"123.0.2420.97\"",
        "sec-ch-ua-full-version-list": "\"Microsoft Edge\";v=\"123.0.2420.97\", \"Not:A-Brand\";v=\"8.0.0.0\", \"Chromium\";v=\"123.0.6312.123\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": "\"\"",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-ch-ua-platform-version": "\"10.0.0\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "sec-ms-gec": "B8B676248A2877EE71418C1F4ABBC5156F93617371372D3A17B831C7F31D6795",
        "sec-ms-gec-version": "1-123.0.2420.97",
        "sec-ms-inbox-fonts": "Roboto",
        "upgrade-insecure-requests": "1",
        "x-client-data": "eyIxIjoiMCIsIjEwIjoiIiwiMiI6IjAiLCIzIjoiMCIsIjQiOiItNDU5ODE3MTEwMTM2NTQ5MzMyMSIsIjUiOiIiLCI2Ijoic3RhYmxlIiwiNyI6IjExNTk2NDExNjk5MyIsIjkiOiJkZXNrdG9wIn0=",
        "x-edge-shopping-flag": "1",
        "Referer": "https://www.bing.com/",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
    params = {
        'q': query,  
        'hl': 'en',             
        'gl': 'us',             
        'num': '500'             
    }
    try:
        session = HTMLSession()
        response = session.get("https://www.bing.com/search", headers=headers, params=params)

    except Exception as e:
        print(e)
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
        
    cite_tags = soup.find_all('cite')
    cite_contents = [cite.get_text() for cite in cite_tags]
    seed_urls = [x for x in cite_contents if is_crawlable(x) and is_valid_mime_type(x)]
    return seed_urls


def fetch_ecosia_links(query):
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "cookie": "ECUNL=9bc12e96-fdfb-472c-9953-f0b1d5dd5efb; ECCC=eamp; _cfuvid=5WW_gUH4GSZKTOOgjonfZFUgjyT39FK7ysxx2lhcFYo-1713426858642-0.0.1.1-604800000; ECFG=a=1:as=1:cs=0:dt=pc:f=i:fr=0:fs=1:l=de:lt=1713431578553:mc=de-de:nf=1:nt=0:pz=0:t=391:tt=0:tu=auto:wu=auto:ma=1; __cf_bm=LmKEA6VSe2iyPtDM5YAX2QRbASmHVpOsoVvlrHPasuE-1713431908-1.0.1.1-G0FDuHW95R8aU8Dvjf8N17ZKTUo4m6MEw7FYWckqmh2K2_DiYSkMkJjSMmJCpO.o1mDyImQlI9Rv61vZpCIQ8g",
        "dnt": "1",
        "referer": "https://www.ecosia.org/",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"123\", \"Not:A-Brand\";v=\"8\", \"Chromium\";v=\"123\"",
        "sec-ch-ua-arch": "x86",
        "sec-ch-ua-bitness": "64",
        "sec-ch-ua-full-version": "123.0.2420.97",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": "\"\"",
        "sec-ch-ua-platform": "Windows",
        "sec-ch-ua-platform-version": "10.0.0",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
    }
    params = {
        'q': query,
        'method': 'index',
        'hl': 'en',             
        'gl': 'us',             
        'num': '500'             
    }
    try:
        session = HTMLSession()
        response = session.get("https://www.ecosia.org/search?method=index&q=earth%20observation", headers=headers)

    except Exception as e:
        print(e)
        return None
    
    with open("ecosia.html", "w+") as file:
        file.write(response.text)

    return 


def fetch_openalex_links(query):
    base_url = "https://api.openalex.org"

    def get_concept_id(query):
        response = requests.get(f"{base_url}/concepts", params={"filter": f"display_name.search:{query}"})
        data = response.json()
        for concept in data['results']:
            if query.lower() in concept['display_name'].lower():
                return concept['id']
        return None

    def get_works_by_concept(concept_id):
        response = requests.get(f"{base_url}/works", params={"filter": f"concepts.id:{concept_id}"})
        return response.json()

    concept_id = get_concept_id(query)
    if concept_id:
        works = get_works_by_concept(concept_id)    
        results = works["results"]
        links = []
        for result in results:
            links.append(result["doi"])
        return links
    else:
        print(f"Concept ID not found for {query}")


def fetch_doaj_links(query):
    url = f"https://doaj.org/api/search/articles/{query.replace(' ', '%20')}?pageSize=100"
    response = requests.get(url)
    results = response.json()["results"]
    with open("doaj.json", "w+") as file:
        json.dump(response.json(), file, indent=4)
    links = []
    for result in results:
        if "bibjson" in result:
            languages = result["bibjson"]["journal"]["language"]
            if len(languages) > 1:
                continue
            links.append(result["bibjson"]["link"][0]["url"])
    return links

def get_reference_corpus(urls, chunked=True):
    def chunk_up_text(text, chunk_size=512):
        split_text = text.split(" ")
        chunked_text = [" ".join(split_text[i:i+chunk_size]) for i in range(0, len(split_text), chunk_size)]
        return chunked_text

    def clean_text(text):
        #remove urls
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S*\.com\S*', '', text)
        text = re.sub(r'\S*www\.\S*', '', text)

        #remove citation
        text = re.sub(r'\([\w\s,.&;-]+(\d{4})\)', '', text)

        #remove special chars
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        #remove latex stuff
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\$\$(.*?)\$\$', '', text)
        text = re.sub(r'\\\(.*?\\\)', '', text)
        text = re.sub(r'\\\[(.*?)\\\]', '', text)

        #remove figure/table headers
        text = re.sub(r'(Figure|Table|FIGURE|TABLE|Fig|Tab) \d+', '', text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    extractor = extractors.ArticleExtractor()
    raw_documents = []
    for url in urls:
        try:
            document = extractor.get_doc_from_url(url)
            text = document.content
            blocks = document.text_blocks
        except:
            print(f"not allowed to crawl {url}")
            continue
        raw_documents.append(text)
    
    documents = [clean_text(doc) for doc in raw_documents]
    if chunked:
        chunked_document = []
        chunk_size = 512
        for doc in documents:
            text_chunks = chunk_up_text(doc)
            for text in text_chunks:
                if len(text) > chunk_size//2:
                    chunked_document.append(text)
        return chunked_document
    return documents

def get_bert_text_embedding(document, cache_file="bert_embeddings_webdata.json"):
    doc_hash = hashlib.sha256(document.encode('utf-8')).hexdigest()
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    if doc_hash in cache:
        cached_embedding = torch.tensor(cache[doc_hash])
        return cached_embedding
    else:
        inputs = bert_tokenizer(document, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)
        
        with open(cache_file, 'w') as f:
            cache[doc_hash] = embeddings.detach().numpy().tolist()
            json.dump(cache, f)
        
        return embeddings
    

def execute_HITS(seed_urls, cached_graph="cache.json", max_iter=20, N=10, depth=0):
    def fetch_url(url, session):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            links = {urlparse(link).geturl() for link in response.html.absolute_links}
            return url, links
        except httpx.RequestError as e:
            print(f"Error fetching {url}: {str(e)}")
            return url, set()
        
    def create_web_graph(seed_urls):
        if os.path.exists(cached_graph):
            with open(cached_graph, 'r') as file:
                data = json.load(file)
            graph = {k: set(v) for k, v in data.items()}
        else:
            graph = defaultdict(set)
            for i in range(depth+1):
                print(f"Depth {i}; Crawling: {len(seed_urls)}")
                with HTMLSession() as session:
                    with ThreadPoolExecutor(max_workers=50) as executor:
                        future_to_url = {executor.submit(fetch_url, url, session):url for url in seed_urls}
                        child_urls = []
                        for future in tqdm(as_completed(future_to_url)):
                            url = future_to_url[future]
                            try:
                                url, links = future.result()
                                graph[url] |= links
                                child_urls.extend(links)
                            except Exception as e:
                                print(f"Error processing {url}: {str(e)}")

                seed_urls = child_urls
            json_graph = {k: list(v) for k, v in graph.items()}
            with open(cached_graph, 'w+') as file:
                json.dump(json_graph, file, indent=4)
        return graph
    
    graph = create_web_graph(seed_urls)

    pages = list(graph.keys())
    num_pages = len(pages)
    
    hubs = {page: 1.0 for page in pages}
    authorities = {page: 1.0 for page in pages}
    
    for _ in range(max_iter):
        old_hubs = hubs.copy()
        old_authorities = authorities.copy()
        
        authority_updates = Counter()
        for page in pages:
            for linked_page in graph[page]:
                if linked_page in authorities:
                    authority_updates[linked_page] += old_hubs[page]
        
        for page in pages:
            authorities[page] = authority_updates[page]

        hub_updates = Counter()
        for page in pages:
            for linked_page in graph[page]:
                if linked_page in authorities:
                    hub_updates[page] += authorities[linked_page]

        for page in pages:
            hubs[page] = hub_updates[page]

        norm = sum(authorities.values())**0.5
        authorities = {page: val / norm for page, val in authorities.items()}
        
        norm = sum(hubs.values())**0.5
        hubs = {page: val / norm for page, val in hubs.items()}


    top_n_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:N]
    top_n_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:N]
    

    # print("TOP HUBS", "@"*50)
    # print(top_n_hubs)

    # print("TOP AUTHS", "@"*50)
    # print(top_n_authorities)

    return hubs, authorities, top_n_hubs, top_n_authorities


def main():
    query = "earth observation"
    if os.path.exists(f"seeds_{query.replace(' ', '_')}.txt"):
        with open(f"seeds_{query.replace(' ', '_')}.txt", "r") as file:
            seed_urls = [url.strip() for url in file]
    else:
        potential_urls = []
        potential_urls.extend(fetch_doaj_links(query))
        potential_urls.extend(fetch_openalex_links(query))
        potential_urls.extend(fetch_search_engine_seeds(query))
        potential_urls = [url for url in tqdm(potential_urls) if is_crawlable(url)]
        seed_urls = set(potential_urls)
        with open(f"seeds_{query.replace(' ', '_')}.txt", "w+") as file:
            for link in seed_urls:
                file.write(link+"\n")
    print("Num Seeds:", len(seed_urls))
    depth = 0
    hub, auth, top_hub, top_auth = execute_HITS(seed_urls, cached_graph=f"graph_{query.replace(' ', '_')}_depth_{depth}.json", depth=depth, N=250)
    
    print(len(hub), len(auth), len(top_hub), len(top_auth))
    with open(f"seeds_depth_{depth}.txt", "w+") as file:
        seed_urls = [x[0] for x in top_auth] + [x[0] for x in top_hub]
        for seed in seed_urls:
            file.write(seed+"\n")

    corpus_urls = [x[0] for x in top_auth] if depth > 0 else seed_urls

    reference_documents = get_reference_corpus(corpus_urls)
    with open(f"reference_documents_{query.replace(' ', '_')}.txt", "w+") as file:
        for doc in reference_documents:
            file.write(doc+"\n")

    with h5py.File(f"../Embedding/data/corpus_embedding_d_{depth}.hdf5", 'w') as hdf5_file:
        for idx, doc in enumerate(tqdm(reference_documents)):
            embedding = get_bert_text_embedding(doc).detach().numpy()
            hdf5_file.create_dataset(f'embedding_{idx}', data=embedding, compression='gzip')

if __name__ == "__main__":
    main()

# curl ^"https://www.google.de/search?q=climate+change&sca_esv=c2c122aff106e202&sxsrf=ACQVn0-qSQEPix_bvYF1Ie5VnFXhD50sGw^%^3A1713429799193&source=hp&ei=J90gZuzmCNDxi-gPh4SFmAM&iflsig=ANes7DEAAAAAZiDrN1ARjp77Y4LbL0iorwKSl9W5xl3I&ved=0ahUKEwislpq0r8uFAxXQ-AIHHQdCATMQ4dUDCBc&uact=5&oq=climate+change&gs_lp=Egdnd3Mtd2l6Ig5jbGltYXRlIGNoYW5nZTIKECMYgAQYJxiKBTIKEAAYgAQYQxiKBTIKEAAYgAQYQxiKBTIKEAAYgAQYQxiKBTIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAESKIiUIQIWN0UcAF4AJABAJgBXaAB_QaqAQIxNLgBA8gBAPgBAZgCD6AC4weoAgrCAgcQIxgnGOoCwgILEC4YgAQYxwEYrwHCAgUQLhiABMICCxAuGIAEGNEDGMcBwgIIEC4YgAQY1ALCAgoQABiABBgUGIcCmAMNkgcCMTWgB7Zx&sclient=gws-wiz^" ^
#   -H "accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" ^
#   -H "accept-language: de,de-DE;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6" ^
#   -H "cookie: SID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnV_He2m6ZCg7iR02oHDRqPewACgYKAYkSAQASFQHGX2MiJYIvsk_W3TwQd6hkGyoXkRoVAUF8yKrxv4be_c6KgoeoPL7LgenE0076; __Secure-1PSID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnVeC-P0B2SDEl9SwDuQ1oMSwACgYKAT4SAQASFQHGX2MidKMmb45almtnnSZjHZj8qBoVAUF8yKqrsDEtenJokob51hABeKGk0076; __Secure-3PSID=g.a000iAg1jT-tmYrDvp9IMRcJR2lrgO2uqPm-PDa-PF0z4zgQ6KnVZu6DHG5QsisGTUGQJ95-mQACgYKAb0SAQASFQHGX2MihLJVP806UgrORtDNhbvMIRoVAUF8yKpCr3qW5WKaoNADtEKMV9-g0076; HSID=A9quHzDj7YtkvFXZf; SSID=AfA-_ClZswKnBQu5a; APISID=fCwuPizlA06PC3Lh/AAIv18dK4y-njgHY7; SAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; __Secure-1PAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; __Secure-3PAPISID=Xwi6djrm6aXsLrcW/ATe8twAp-Eqku9GPU; NID=512=SXob5RvKanDNXYpfZ_lJIwbPcA-Ri2GBFeEpNY6kpC1rNsp-RY9PW9CQbffzJW1hbtK-v2PhOwIg61B9sveljNBq6ssETXkEaCvnO0gZDLLP9tOnzfpinnj6aqRIwPg2SydQ2k046pDv0OeLo3WHVVlY3Fh2Q88Zd8IpL-3qDtwh338uhr4P4fuv0xtZI-9t9_qSwEgYAStPTKBHDwXQNkPy1SfC3JacWalkbPNldmut5GbSeo7D__oDv1M8TQN-Budxvad1hvs-OlqnHw; AEC=AQTF6Hy7nE_iLLiGVNZyPoQ29N26OCVAeMStfphG6v5SPZLAETw_nvmcMg; __Secure-ENID=19.SE=QjOlIRWLXy-fJswsAaHbBf0WY_ZW-wonXTvx5HkLluiUIh91N1PyI_TOV-a5BGO5y3HqymYFTEhcqLmKNwxGsfSBf9CSGrZSGEfcPUPnX0YzPgqJIE9gjEswAG8ObjhIse6tBowKOMwVedcjvaOaZp0dRyoVyEamnI9liHSyIbynUc-9ygHS_pqaVOE180nyhC6Y7AMeashAKwh5J1XK7KSIEnZ7Q5XxMdQ1pj-GZxty8-7W-IvTHkCVUB48YmVwkwze0zcImdle_G2MPsyvn8c; OGPC=19041161-1:19008535-1:19026797-2:; OGP=-19008535:-19026797:" ^
#   -H "dnt: 1" ^
#   -H "referer: https://www.google.de/" ^
#   -H ^"sec-ch-ua: ^\^"Microsoft Edge^\^";v=^\^"123^\^", ^\^"Not:A-Brand^\^";v=^\^"8^\^", ^\^"Chromium^\^";v=^\^"123^\^"^" ^
#   -H ^"sec-ch-ua-arch: ^\^"x86^\^"^" ^
#   -H ^"sec-ch-ua-bitness: ^\^"64^\^"^" ^
#   -H ^"sec-ch-ua-full-version: ^\^"123.0.2420.97^\^"^" ^
#   -H ^"sec-ch-ua-full-version-list: ^\^"Microsoft Edge^\^";v=^\^"123.0.2420.97^\^", ^\^"Not:A-Brand^\^";v=^\^"8.0.0.0^\^", ^\^"Chromium^\^";v=^\^"123.0.6312.123^\^"^" ^
#   -H "sec-ch-ua-mobile: ?0" ^
#   -H ^"sec-ch-ua-model: ^\^"^\^"^" ^
#   -H ^"sec-ch-ua-platform: ^\^"Windows^\^"^" ^
#   -H ^"sec-ch-ua-platform-version: ^\^"10.0.0^\^"^" ^
#   -H "sec-ch-ua-wow64: ?0" ^
#   -H "sec-fetch-dest: document" ^
#   -H "sec-fetch-mode: navigate" ^
#   -H "sec-fetch-site: same-origin" 
#   -H "sec-fetch-user: ?1" ^
#   -H "upgrade-insecure-requests: 1" ^
#   -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"