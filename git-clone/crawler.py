from lxml import html
import re
import requests
from bs4 import *
from time import sleep
class Crawler:
    def __init__(self):
        self.p = 1
        self.index = 0
        self.q = ['inference pipeline', 'seldon core', 'raytune','inference sagemaker']
        self.base_urls = [f'https://github.com/search?p={self.p}&q={self.q[0]}&type=Repositories']
        self.urls = []



    def download_links(self, response, idx):
        # //*[@id="js-pjax-container"]/div/div[3]/div/ul/li[1]/div[2]/div[1]/div/a
        # //*[@id="js-pjax-container"]/div/div[3]/div/ul/li[2]/div[2]/div[1]/div/a
        # //*[@id="js-pjax-container"]/div/div[3]/div/ul/li[3]/div[2]/div[1]/div/a

        url = response.xpath(f'//*[@id="js-pjax-container"]/div/div[3]/div/ul/li[{idx}]/div[2]/div[1]/div/a')
        return url[0].attrib['href']

    def data_getter(self, url):
        page = requests.get(url)
         # get all the text in the li tag
        if page.status_code == 429:
            print("ssss")
        data = html.fromstring(page.content)
        list_repo = data.xpath('//*[@id="js-pjax-container"]/div/div[3]/div/ul')  # loop through the ul tag, whose child tag contains class attribute and the value is 'name'
        if len(list_repo) == 0:
            return None, -1
        len_urls = list_repo[0].xpath('li').__len__()
        return data, len_urls

    def url_builder(self):
        return f'https://github.com/search?p={self.p}&q={self.q[self.index]}&type=Repositories'
    
    def request(self):
        sleep(10)
        page = requests.get(self.base_urls[0])
         # get all the text in the li tag
        data = html.fromstring(page.content)
        list_repo = data.xpath('//*[@id="js-pjax-container"]/div/div[3]/div/ul')  # loop through the ul tag, whose child tag contains class attribute and the value is 'name'
        len_urls = list_repo[0].xpath('li').__len__()
        for i in self.base_urls:
            while len_urls != 0:
                for i in range(0,len_urls):
                    self.urls.append(self.download_links(data, i + 1))
                self.p += 1
                url = self.url_builder()
                sleep(7)
                data, len_urls = self.data_getter(url)
            self.index += 1
            url = self.url_builder()
            sleep(7)
            data, len_urls = self.data_getter(url)
        return self.urls

    def find_most_important(self, urls, word):
        for url in urls:
            new_url = f'https://github.com'+url +'/search?q={word}'
            data, len_parts = self.data_getter(new_url)
            if len_parts > 0 :
                print(new_url)
    def download_images(self, urls):
<<<<<<< HEAD
<<<<<<< HEAD
        f1 = open("reses.txt",a)
=======
>>>>>>> 2c730123a2996da74dd0bd978fd20cf37b5f2aa3
=======
>>>>>>> 2c730123a2996da74dd0bd978fd20cf37b5f2aa3
        for url in urls:
            print(f'processing {url}...')
            response = requests.get('https://github.com'+url)
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tags = soup.find_all('img')
            images = [img['src'] for img in image_tags]
            for img in images:
                filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', img)
                if not filename:
                    print("Regular expression didn't match with the url: {}".format(img))
                    continue
<<<<<<< HEAD
<<<<<<< HEAD
                with open('images/'+filename.group(1), 'wb') as f:
                    if 'http' not in img:
                        img = '{}{}'.format('https://github.com'+url, img)
                    response = requests.get(img)
                    f1.write(url + " : "+ filename.group(1))
=======
=======
>>>>>>> 2c730123a2996da74dd0bd978fd20cf37b5f2aa3
                with open(filename.group(1), 'wb') as f:
                    if 'http' not in img:
                        img = '{}{}'.format('https://github.com'+url, img)
                    response = requests.get(img)
<<<<<<< HEAD
>>>>>>> 2c730123a2996da74dd0bd978fd20cf37b5f2aa3
=======
>>>>>>> 2c730123a2996da74dd0bd978fd20cf37b5f2aa3
                    f.write(response.content)
            print("Download complete, downloaded images can be found in current directory!")

crawler = Crawler()
urls = crawler.request()
crawler.download_images(urls)


from pydriller import Repository
for commit in Repository('https://github.com'+ urls[0]).traverse_commits():
    print(commit.msg)
    print(commit.author.name)
    print()






