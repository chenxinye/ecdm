# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:54:33 2019

@author: chenxinye
"""


def text_divided(readfile, n, encode = 'UTF-8'):
    
    f = open(readfile,'r',encoding=encode)
    lines = f.readlines()
    length = len(lines)
    
    nlength = round(length/n)
    
    for i in range(n):
        filname = '_part' + str(i + 1) + '.txt'
        with open(filname,'w',encoding=encode) as fw:
            fw.write("".join(lines[i*nlength : (i+1)*nlength]))
    
    f.close()
    print('done!')
    
text_divided(readfile = 'test.txt', n = 3)