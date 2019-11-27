# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:33:42 2019
Licensed under the terms of the MIT License

@describe: The program is to get customers reviewS in JD.com. 
           Just a test, commercial use is forbidden.
@author: chenxinye

"""

import re
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm
print(sys.path)

class get_JDreview:
    def __init__ (self, sku, page, nature, save = True):
        self.review_nature = {'好评':3, '中评':2, '差评':1}
        self.header = {'Referer': 'https://item.jd.com/' + str(sku) +'.html',
                       'Sec-Fetch-Mode': 'no-cors',
                       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'
                      }

        session = requests.Session()
        response = list()

        for npage in tqdm(range(page)):
            print("\n page: ", npage)
            i = 1; condition = True
            while(condition):
                try:
                    self.url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv11883&productId=' \
                               + str(sku) + '&score=' + str(self.review_nature[nature]) + '&sortType=5&page='+ str(npage) + '&pageSize=10&isShadowSku=0&rid=0&fold=1'

                    html = session.get(self.url, headers=self.header)
                    response += self.research_transform('"content":".*?",["vcontent"|"id"]', html.text)
                    time.sleep(4)
                    condition = False

                except:
                    i = i + 1
                    if i == 5:
                        print("wrong! page:", npage)
                        break

        self.df = pd.DataFrame({'comments':list(set(response))})

        if save:
            self.df.to_csv(str(sku) + nature +  '.csv', encoding='utf_8_sig', index = False)


    def research_transform(self, regex, text):

        res = re.findall(regex, text)
        contents = list()
        for r in res:
            content = re.findall(u"[\u4e00-\u9fa5，：]+",r)
            content = ' '.join(content)
            print(content)
            contents.append(content)

        return contents

if __name__ == '__main__':
    sku_collect = [
            '7824307',
            '136360',
            '7265743',
            '46472869374',
            '100000198663',
            '51382064682',
            '3567887'
            ]
    
    for sku in sku_collect:
        objn = get_JDreview(sku = sku, page = 80, nature = 'ne', save = True)
        objp = get_JDreview(sku = sku, page = 80, nature = 'po', save = True)