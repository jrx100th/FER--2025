from icrawler.builtin import GoogleImageCrawler

def download_images(query, download_dir=r"C:\Users\jenit\Downloads\scraping\angry human face expression"):
    crawler = GoogleImageCrawler(storage={'root_dir': download_dir})
    crawler.crawl(keyword=query, max_num=100)
    print(f"Images downloaded to: {download_dir}")


if __name__ == "__main__":
    download_images("happy image human face expression")
