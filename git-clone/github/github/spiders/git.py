import scrapy


class GitSpider(scrapy.Spider):
    name = 'git'
    allowed_domains = ['gihub.com']
    start_urls = ['http://gihub.com/']

    def parse(self, response):
        pass
