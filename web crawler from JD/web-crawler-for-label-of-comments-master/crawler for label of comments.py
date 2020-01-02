# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:34:35 2019
Licensed under the terms of the MIT License

@describe: The program is to get consumers comment tags in JD.com. 
           Just a test, commercial use is forbidden.

@author: chenxinye

"""

import sys
print(sys.path)
import re
import requests
from urllib import request
import time
import pandas as pd



class GET_REVIEW:
    def __init__(self, _skuID, name = 'Undefined', save = True):
        self.review = self.get_review(_skuID, name, save)


    def get_review(self, _skuID, name, save):
        header = {
            'authority': 'sclub.jd.com',
            'method': 'GET',
            'path': '/comment/productPageComments.action?callback=fetchJSON_comment98vv7129&productId=1178676&score=0&sortType=5&page=1&pageSize=10&isShadowSku=0&rid=096fe6a400c7dcba&fold=1',
            'scheme': 'https',
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cookie': 'areaId=19; ipLoc-djd=19-1607-3155-0; PCSYCityID=CN_440000_440300_440305; shshshfpa=aa61bedf-c365-0f2c-4d14-556a776a2d4b-1571208617; shshshfpb=wwGFCWITOxX6CDrdAYJbdGg%3D%3D; user-key=630d7f40-34ba-4c75-babd-f329b3b77789; cn=0; TrackID=1Dbu4DMC3xq8kpT_b6X-som2WOcsbkXaksbq3glL9XGpN18TLbMFUiNbDEqG0766KdP9KQDX4-hEKz6cbrMfcmmJ_tcnRtunaprMnJExKSuk; pinId=ptD77A39qERA9AXplivx1A; _tp=n2Nd1fLZaYX0imx7vt9QPL8eey4plQk3MsMKQ9tb%2Bn4%3D; _pst=%E5%8D%B0%E6%A9%99007; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|a7fe4217debc4b01983642ac9a22d19d_0_26eeaeba489d4afc8f498ff084f9b8a4|1571617053366; __jdu=156807888393155578223; __jda=122270672.156807888393155578223.1568078884.1571605360.1571617053.18; __jdc=122270672; 3AB9D23F7A4B3C9B=QUHORZ6QAH5KWPJ5ODPQRGO7QJZ2FVCFUODURSEKG4LUV3PNTYBX7GMYA3FRJXGQ6YGLA3BYTF4RKCZ3Y4WCAUAVVI; unpl=V2_ZzNsbUVVFEIiChJVZxhaBGEfFw9EU19FIggRSHobWAc1VEAJEQAQR2lJKFRzEVQZJkB8XUtfRgklTShUehhaAWAzEVxCX18VdBRFVGoYVQ5nCxlZRWdDJXUJR1V6GloGbgMibXJXQSV0OEZQfBBdA24KG19KVUMRcQxAXXgaXjVkUEVZSlIXFWkITlIuBVkHZVQODREHRwkgCUUGcxALV28LGl5yUUU%3D; shshshfp=beb6c65a1bb9c98e1ece5f914f292fab; __jdb=122270672.6.156807888393155578223|18.1571617053; shshshsID=787b21f740d8de824ea4c6c358362d08_6_1571617357337; JSESSIONID=3B3CF67DC2B07492CE3C5B6EA998A5E7.s1',
            'referer': 'https://item.jd.com/1178676.html',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
            }
        
        skuID = list()
        labelID = list()
        averagescoreID = list()
        commentCountID = list()
        
        for sku in _skuID:
            proxy = {'http':'183.60.141.1:443'}
            scoreID =  0
            skuID.append(sku)
            proxyHeader = request.ProxyHandler(proxy)
            opener = request.build_opener(proxyHeader)
            request.install_opener(opener)
            
            api_ = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv7129&productId=' +str(sku) + '&score=' + str(scoreID) + '&sortType=5&page=0&pageSize=10&isShadowSku=0&rid=096fe6a400c7dcba&fold=1'
            
            try:
                response = requests.get(api_, headers = header)
                _text = response.text
                contents = re.findall('"id":.*?,"name":.*?,"count":[0-9]+', _text)
                averagescoreID.append(re.findall('"averageScore":[0-9]', _text)[0].split(":")[1])
                commentCountID.append(re.findall('"commentCount":[0-9]+', _text)[0].split(":")[1])
                
                labelid = ''
                for i in range(len(contents)):
                    label = contents[i].split(":")[2].split(",")[0]
                    numcount = contents[i].split(":")[3]
                    labelid += (label + ':' + numcount).replace('"', '') + ' '
                    
                labelID.append(labelid)
                time.sleep(2)
                
            except:
                try:
                    response = requests.get(api_, headers = header)
                    _text = response.text
                    contents = re.findall('"id":.*?,"name":.*?,"count":[0-9]+', _text)
                    averagescoreID.append(re.findall('"averageScore":[0-9]', _text))
                    commentCountID.append(re.findall('"commentCount":[0-9]+', _text))
                    labelid = ''
                    for i in range(len(contents)):
                        label = contents[i].split(":")[2].split(",")[0]
                        numcount = contents[i].split(":")[3]
                        labelid += (label + ':' + numcount).replace('"', '') + ' '
                    labelID.append(labelid)
                    time.sleep(2)
                    
                except:
                    time.sleep(7)
        
            
        df = pd.DataFrame({'skuID':skuID, 
                           'averagescore':averagescoreID,
                           'commentCount':commentCountID,
                           'labelID':labelID
                           })
        #save = True
        if save:
            df.to_csv(name + '用户评论标签.csv', encoding='utf_8_sig', index = False)
            
        return df
    
    
if __name__ == '__main__':
    
    name = '合生元（BIOSTIME）儿童益生菌粉(益生元)奶味48袋装（0-7岁宝宝婴儿幼儿 法国进口菌粉 活性益生菌 ）'
    _skuID = [
            '1338005',
            '1178676',
            '1178664',
            '1178665',
            '1178668',
            ]
    
    obj = GET_REVIEW(_skuID, name, save = True)





