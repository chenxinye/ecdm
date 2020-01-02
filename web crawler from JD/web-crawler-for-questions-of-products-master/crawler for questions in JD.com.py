# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:48:58 2019

@describe: The program is to get customers questions and corresponding answers according to the given sku of the product in JD.com.
           Just a test, commercial use is forbidden.

@author: chenxinye
"""


import re
import time
import random
import requests
from tqdm import tqdm
import pandas as pd

class crawl_review:
    def __init__(self, sku, page = 15, save = True):
        self.df = self.get_review(sku, page, save)

    def get_review(self, sku, page, save):
        
        header = {
            'authority': 'sclub.jd.com',
            'method': 'GET',
            'path': '/comment/productPageComments.action?callback=fetchJSON_comment98vv7129&productId='+ str(sku) + '&score=0&sortType=5&page=1&pageSize=10&isShadowSku=0&rid=096fe6a400c7dcba&fold=1',
            'scheme': 'https',
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cookie': 'areaId=19; ipLoc-djd=19-1607-3155-0; PCSYCityID=CN_440000_440300_440305; shshshfpa=aa61bedf-c365-0f2c-4d14-556a776a2d4b-1571208617; shshshfpb=wwGFCWITOxX6CDrdAYJbdGg%3D%3D; user-key=630d7f40-34ba-4c75-babd-f329b3b77789; cn=0; TrackID=1Dbu4DMC3xq8kpT_b6X-som2WOcsbkXaksbq3glL9XGpN18TLbMFUiNbDEqG0766KdP9KQDX4-hEKz6cbrMfcmmJ_tcnRtunaprMnJExKSuk; pinId=ptD77A39qERA9AXplivx1A; _tp=n2Nd1fLZaYX0imx7vt9QPL8eey4plQk3MsMKQ9tb%2Bn4%3D; _pst=%E5%8D%B0%E6%A9%99007; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|a7fe4217debc4b01983642ac9a22d19d_0_26eeaeba489d4afc8f498ff084f9b8a4|1571617053366; __jdu=156807888393155578223; __jda=122270672.156807888393155578223.1568078884.1571605360.1571617053.18; __jdc=122270672; 3AB9D23F7A4B3C9B=QUHORZ6QAH5KWPJ5ODPQRGO7QJZ2FVCFUODURSEKG4LUV3PNTYBX7GMYA3FRJXGQ6YGLA3BYTF4RKCZ3Y4WCAUAVVI; unpl=V2_ZzNsbUVVFEIiChJVZxhaBGEfFw9EU19FIggRSHobWAc1VEAJEQAQR2lJKFRzEVQZJkB8XUtfRgklTShUehhaAWAzEVxCX18VdBRFVGoYVQ5nCxlZRWdDJXUJR1V6GloGbgMibXJXQSV0OEZQfBBdA24KG19KVUMRcQxAXXgaXjVkUEVZSlIXFWkITlIuBVkHZVQODREHRwkgCUUGcxALV28LGl5yUUU%3D; shshshfp=beb6c65a1bb9c98e1ece5f914f292fab; __jdb=122270672.6.156807888393155578223|18.1571617053; shshshsID=787b21f740d8de824ea4c6c358362d08_6_1571617357337; JSESSIONID=3B3CF67DC2B07492CE3C5B6EA998A5E7.s1',
            'referer': 'https://item.jd.com/' + str(sku) +'.html',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
        }
    
        ask_contentID = list()
        lastAnswerTimeID = list()
        
        reply_contentID = list()
        replycreatedTimeID = list()
        replymodifiedTimeID = list()
        
        for npage in tqdm(range(page)):
            try:
                wp = 'https://question.jd.com/question/getQuestionAnswerList.action?callback=jQuery' + str(sku) + '6283842&page=' + str(npage) + '&productId=' + str(sku)
                response = requests.get(wp, headers = header)
                text = response.text
                
                reply_date = re.findall('"created":"[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+","modified":"[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+"', text)
                review = re.findall('"content":".*?".*?"created":"[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+","modified":"[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+"', text)
                
                for i in range(len(review)):
                    ask = '"content":".*?","clientType":.*?,"productId":.*?,"status":.*?,"lastAnswerTime":"[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+"'
                    ask_get = re.findall(ask, review[i])
                    try:
                        ask_content = re.findall('"content":".*?"', ask_get[0])[0].split(':')[1].replace('"', '')
                        lastAnswerTime = re.findall('"lastAnswerTime":".*?"', ask_get[0])[0].split('"')[3].replace('"', '')
                    except:
                        continue
                    try:
                        reply = '"content":".*?","systemId"'
                        reply_get = re.findall(reply, review[i])
                        reply_content = re.findall('"content":".*?"', reply_get[0])[1].split(":")[1].replace('"', '')
                        replycreatedTime = reply_date[i].split(',')[0].replace('"created":', '').replace('"', '')
                        replymodifiedTime = reply_date[i].split(',')[1].replace('"modified":', '').replace('"', '')
                    except:
                        continue

                    ask_contentID.append(ask_content)
                    lastAnswerTimeID.append(lastAnswerTime)
                    
                    reply_contentID.append(reply_content)
                    replycreatedTimeID.append(replycreatedTime)
                    replymodifiedTimeID.append(replymodifiedTime)
                
                time.sleep(random.randint(1, 3))
                
            except:
                print("wrong!")
                
        df = pd.DataFrame({'ask_content':ask_contentID, 
                           'lastAnswerTime':lastAnswerTimeID,
                           'reply_content':reply_contentID,
                           'replycreatedTime':replycreatedTimeID,
                           'replymodifiedTime':replymodifiedTimeID
                           })
            
        if save == True:
            df.to_csv(str(sku) + '_qreviews.csv', encoding='utf_8_sig', index = False)
        return df


if __name__ == '__main__':
    #sku = 8231902
    #df = crawl_review(sku, page = 70, save = True)
    
    with open('sku.txt') as f:
        content = f.readlines()
        content = ''.join(content)
        content = content.split(',')
        for sku in content:
            df = crawl_review(sku, page = 35, save = True)