# -*- coding: utf8 -*-
import requests
import sys, time, json

url = 'http://localhost:8888/'

dialog_list = {'str1': '拉娜·德雷这个人你听说过吗？', 'str2': '她是很有名气的一位女歌手。', 'str3': '是啊，我特别喜欢她。'}
dialog_list = json.dumps(dialog_list)
response = requests.put(url, data={'dialog_list': dialog_list})
obj = response.json()
print(obj)
