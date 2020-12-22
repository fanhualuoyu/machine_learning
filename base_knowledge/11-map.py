dict1 = {'李宁':'一切皆有可能','Nike':'Just do It','阿迪达斯':'Impossible is nothing'}
print('李宁的口号是：',dict1['李宁'])

dict = {}
dict2 =  dict.fromkeys({1,2,3,4,5},'number')
print(dict2)

dict2.pop(1)
print(dict2)

dict2.setdefault('小白')
print(dict2)

b={'小白':'狗'}
dict2.update(b)
print(dict2)