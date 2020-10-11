#!/usr/bin/python3

# 1. get img urls
# 2. aiohttp + aiofiles

import os
import re
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
}

support_webs = {
    "hanascan": "https://hanascan.com/",
    "manhuagui": "https://www.manhuagui.com/",
    }

async def get_url_data(url, req_type='text'):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                if req_type == 'bytes':
                    content = await resp.read()
                else:
                    content = await resp.text()
                return content

async def get_chapter_url(web, url):
    content = await get_url_data(url)
    soup = BeautifulSoup(content, 'lxml')
    
    if web == "hanascan":
        chapters = soup.find("div", attrs={"class": "list-wrap"}).find_all("a")
    elif web == "manhuagui":
        pass
    chapters = [chapter['href'] for chapter in chapters][::-1]
    return chapters

async def get_website_name(chapter_url):
    not_support = True
    for web, web_url in support_webs.items():
        if chapter_url.startswith(web_url):
            return web
    if not_support:
        assert False, f"\n\twebsite: {chapter_url} is not in support webs!!"

class ImgDownloader():
    def __init__(self, web, url, semaphore):
        self.chapter_url = url
        self.root = "imgs"
        self.img_urls = []
        self.manga_path = None
        self.semaphore = semaphore
        self.web = web

    async def download(self):
        is_exists = await self.make_directory()
        if is_exists:
            print(f"{self.manga_path} is exists...")
        else:
            await self.collect_img_url()
            tasks = []
            for idx, img_url in enumerate(self.img_urls):
                tasks.append(self.download_imgs(idx, img_url))
            await asyncio.gather(*tasks)


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
    def __init__():
        pass


async def run(url):
    if not os.path.exists('./imgs'):
        os.makedirs('./imgs')

    web = await get_website_name(url)
    chapters = await get_chapter_url(web, url)

    tasks = []
    semaphore = asyncio.Semaphore(20)
    for chapter in chapters:
        img_d = ImgDownloader(web, chapter, semaphore)
        tasks.append(img_d.download())

    await asyncio.gather(*tasks)

def main(url):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run(url))
    finally:
        loop.close()

if __name__ == "__main__":
    url = "https://hanacan.com/manga-ayakashi-triangle-raw.html"
    main(url)
