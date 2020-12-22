# 原始字符串
str = r'c:\now'
print(str)

# 跨越多行的字符串
strs = """python真的很有趣，\npython很有用"""
print(strs)

str1 = 'xiaoxie'
print(str1.capitalize())

str2 = "ZHESHIDAXIEBIANXIAOXIE"
print(str2.casefold())

print(str2.center(50))

print(str2.count('Z'))

print(str2.endswith('E'))

print("{0} love {1}.{2}".format("I","python","com"))

print("{a} love {b}.{c}".format(a="I",b="python",c="com"))

print("{{0}}".format("不打印"))

print("{0:.1f}{1}".format(25.652,"GB"))

print('%c'%97)

print('%s' % "I love python")