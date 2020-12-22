language = ['python','java','c#','vue']
language.append('c++')
print(language)

language.extend(['c','javascript'])
print(language)

language.insert(0,'php')
print(language)

language.remove('php')
print(language)

del language[1]
print(language)

name = language.pop()
print(name)
print(language)

language.pop(1)
print(language)

spliceLanuage = language[1:3]
print(spliceLanuage)


list1 = [123,456]
list2 = [234,123]
print(list1 < list2)

list3 = list1 + list2
print(list3)

print(list3*3)

print(123 in list3)

print(list3.count(123))

print(list3.index(123,1,4))

list3.reverse()
print(list3)

list4 = [5,51,4,52,89,100]
list4.sort()
print(list4)

list4.sort(reverse=True)
print(list4)
