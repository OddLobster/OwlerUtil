import requests
import urllib.robotparser
import urllib

from bs4 import BeautifulSoup
from requests_html import HTMLSession
from httpx import Client


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

def get_search_engine_seeds():
    pass

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
        'num': '50'             
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
        'num': '50'             
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

def main():
    query = "earth observation"
    seed_urls = []
    print("Fetching links from Google:")
    google_links = fetch_google_links(query)
    for link in google_links:
        print(link)
        seed_urls.append(link)

    print("\nFetching links from Bing:")
    bing_links = fetch_bing_links(query)
    for link in bing_links:
        print(link)
        seed_urls.append(link)

    # TODO
    # print("\nFetching links from Ecosia:")
    # ecosia_links = fetch_ecosia_links(query)
    # for link in ecosia_links:
    #     print(link)

    

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
#   -H "sec-fetch-site: same-origin" ^
#   -H "sec-fetch-user: ?1" ^
#   -H "upgrade-insecure-requests: 1" ^
#   -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"