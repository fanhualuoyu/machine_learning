#爬虫
#import urllib.request as request
# response = request.urlopen("http://www.baidu.com")
# html = response.read()
# html = html.decode("utf-8")
# print(html)

'''
import urllib.request as request
response = request.urlopen("http://placekitten.com/500/600")
#response.geturl()：访问的路径
#response.info()：请求信息
#response.getcode()：状态码
cat_img = response.read()
with open('cat_500_600.jpg','wb') as f:
  f.write(cat_img)
'''

import urllib.request
import os
import random

def url_open(url):
  req = urllib.request.Request(url)
  req.add_header('User-Agent','Mozilla/5.0 AppleWebKit/537.36')
  proxies = ['119.6.144.70:81','111.1.36.9:80','203.144.144.162:8080']
  proxy = random.choice(proxies)
  proxy_support = urllib.request.ProxyHandler({'http':proxy})
  opener = urllib.request.build_opener(proxy_support)
  urllib.request.install_opener(opener)
  response = urllib.request.urlopen(url)
  html = response.read()
  return html

def get_page(url):
  html = url_open(url).decode('utf-8')
  a = html.find('current-comment-page')+23
  b = html.find(']',a)
  return html[a:b]

def find_imgs(url):
  html = url_open(url).decode('utf-8')
  img_addrs = []
  a = html.find('img src=')
  while a != -1:
    b = html.find('.jpg',a,a+255)
    if b != -1:
      img_addrs.append(html[a+9:b+4])
    else:
      b = a+9
    a = html.find('img src=',b)
  return img_addrs

def save_imgs(folder,img_addrs):
  for each in img_addrs:
    filename = each.split('/')[-1]
    with open(filename,'wb') as f:
      img = url_open(each)
      f.write()
  pass

def download_img(folder = 'ooxx',pages = 10):
  os.mkdir(folder)
  os.chdir(folder)
  url = "http://jandan.net/ooxx/"
  page_num = int(get_page(url))
  for i in range(pages):
    page_num -= i
    page_url = url + 'page-' + str(page_num) + '#comments'
    img_addrs = find_imgs(page_url)
    save_imgs(folder,img_addrs)

if __name__ == "__main__":
    download_img()