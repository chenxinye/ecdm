# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:03:19 2019

This code greatly simplifies a lot of manual work in my work.
 It's simple, but because it's so useful, I think it really means a lot to me. 

@author: chenxinye
"""

def drop_dup(is_):
    """this method is to help you find duplicate features and reset the columns!"""
    ip = list()
    ipn = list()
    for i in list(set(is_)):
        if is_.count(i) >= 2:
            print(i,is_.count(i))
            ip.append(i)
            ipn.append(is_.count(i))
            
    ipnn = [0] * len(ip)
    dic2 = dict(zip(ip,ipnn))
    for i in range(len(is_)):
        fe = is_[i]
        if fe in ip:
            dic2[fe] += 1
            is_[i] = fe + str(dic2[fe])
    return is_