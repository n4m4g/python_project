#!/usr/bin/python3

# 1. get img urls
# 2. aiohttp + aiofiles

import os
import re
import sys
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup

class ImgDownloader():
    def __init__(self):
        self.root = "imgs"
        self.support_webs = {
            "hanascan": "https://hanascan.com/",
            "manhuagui": "https://www.manhuagui.com/",
            }
        self.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
        }

    def create_dir(self, path):
        has_exists = False
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            has_exists = True
        return has_exists

    async def get_url_data(self, url, req_type='text'):
        async with self.semaphore:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        if req_type == 'bytes':
                            content = await resp.read()
                        else:
                            content = await resp.text()
                        return content

    async def download_img(self, chapter_path, idx, url):
        img_ext = re.match(".*\.([a-z]+)", url).group(1)
        img_name = f"{idx:03d}.{img_ext}"
        img_path = os.path.join(chapter_path, img_name)
        if not os.path.exists(img_path):
            try:
                content = await self.get_url_data(url, req_type='bytes')
                async with aiofiles.open(img_path, 'wb') as f:
                    await f.write(content)
                print(f"{img_path}, done...")
            except Exception as e:
                print(f"img_path: {img_path}")
                print(f"url: {url}")
                print(f"Failed since {e}")
        else:
            # print(img_path, "is exists")
            pass


class HanascanDownloader(ImgDownloader):
    def __init__(self, url, semaphore):
        super(HanascanDownloader, self).__init__()
        self.url = url
        self.semaphore = semaphore
        self.web = "hanascan"

    async def download(self):
        tasks = [self.extract_chapter_links(self.url)]
        results = await asyncio.gather(*tasks)
        self.manga_path, chapter_links = results[0]

        tasks = [self.extract_img_links(chapter_link) for chapter_link in chapter_links]
        results = await asyncio.gather(*tasks)

        tasks = []
        for has_exists, chapter_path, img_links in results:
            for idx, img_link in enumerate(img_links):
                tasks.append(self.download_img(chapter_path, idx, img_link))
        await asyncio.gather(*tasks)

    async def extract_chapter_links(self, url):
        content = await self.get_url_data(url)
        soup = BeautifulSoup(content, 'lxml')
        manga_name = soup.find("ul", attrs={"class": "manga-info"}).h1.text
        manga_path = os.path.join(self.root, manga_name)
        _ = self.create_dir(manga_path)
        chapter_links = soup.find("div", attrs={"class": "list-wrap"}).find_all("a")
        chapter_links = [chapter['href'] for chapter in chapter_links][::-1]
        chapter_links = [self.support_webs[self.web]+url for url in chapter_links]
        return (manga_path, chapter_links)

    async def extract_img_links(self, chapter_link):
        content = await self.get_url_data(chapter_link)
        soup = BeautifulSoup(content, 'lxml')
        title = soup.select("div.chapter-img.tieude")[0].find("font").text
        chapter_path = os.path.join(self.manga_path, title)
        has_exists = self.create_dir(chapter_path)
        imgs = soup.find("article", attrs={"id": "content"}).find_all("img")
        img_links = [img["src"] for img in imgs]
        return (has_exists, chapter_path, img_links)

class ManhuaguiDownloader(ImgDownloader):
    def __init__(self):
        super(ManhuaguiDownloader, self).__init__()

    async def make_directory(self):
        is_exists = False
        if self.web == "hanascan":
            manga_name = re.match("read-(.*)\.html", self.chapter_url)
        elif self.web == "manhuagui":
            pass
        self.manga_path = os.path.join(self.root, manga_name.group(1))
        if not os.path.exists(self.manga_path):
            os.makedirs(self.manga_path)
        else:
            is_exists = True
        return is_exists

    async def collect_img_url(self):
        url = support_webs[self.web]+self.chapter_url
        content = await get_url_data(url)
        soup = BeautifulSoup(content, 'lxml')
        if self.web == "hanascan":
            imgs = soup.find("article", attrs={"id": "content"}).find_all("img")
        elif self.web == "manhuagui":
            pass
        self.img_urls = [img["src"] for img in imgs]

    async def download_imgs(self, idx, url):
        async with self.semaphore:
            await asyncio.sleep(0.1)
            content = await get_url_data(url, req_type='bytes')
            img_ext = re.match(".*\.([a-z]+)", url).group(1)
            img_name = f"{idx:03d}.{img_ext}"
            img_path = os.path.join(self.manga_path, img_name)
            async with aiofiles.open(img_path, 'wb') as f:
                await f.write(content)
            print(f"{img_path}, done...")

class WebDownloaderHandler():
    def __init__(self, url, semaphore):
        self.url = url
        self.semaphore = semaphore
        self.support_webs = {
            "hanascan": "https://hanascan.com/",
            "manhuagui": "https://www.manhuagui.com/",
            }

    def get_downloader(self):
        web = self.get_website_name()
        if web == "hanascan":
            downloader = HanascanDownloader(self.url, self.semaphore)
        elif web == "manhuagui":
            downloader = ManhuaguiDownloader(self.url, self.semaphore)
        else:
            print(f"website: {self.url} is not in support webs!!")
            sys.exit()
        return downloader

    def get_website_name(self):
        for web, web_url in self.support_webs.items():
            if self.url.startswith(web_url):
                return web
        return None


async def run(urls):
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    semaphore = asyncio.Semaphore(15)

    for url in urls:
        handler = WebDownloaderHandler(url, semaphore)
        downloader = handler.get_downloader()
        await downloader.download()

def main(urls):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(urls))

if __name__ == "__main__":
    urls = [
        "https://hanascan.com/manga-ayakashi-triangle-raw.html",
        "https://hanascan.com/manga-fukushuu-o-koinegau-saikyou-yuusha-wa-yami-no-chikara-de-senmetsu-musou-suru-raw.html",
        "https://hanascan.com/manga-izure-saikyo-no-renkinjutsu-shi-raw.html",
        "https://hanascan.com/manga-kaifuku-jutsushi-no-yarinaoshi-raw.html",
        "https://hanascan.com/manga-lottery-grand-prize-musou-harem-rights-raw.html",
        "https://hanascan.com/manga-parallel-paradise-raw.html",
        "https://hanascan.com/manga-queens-quality-raw.html",
        "https://hanascan.com/manga-tensei-shitara-dai-nana-ouji-dattanode-kimamani-majutsu-o-kiwamemasu-raw.html",
        "https://hanascan.com/manga-the-reincarnation-magician-of-the-inferior-eyes-raw.html",
        "https://hanascan.com/manga-tsugumomo-raw.html",
        "https://hanascan.com/manga-dokyuu-hentai-hxeros-raw.html",
    ]

    main(urls)
