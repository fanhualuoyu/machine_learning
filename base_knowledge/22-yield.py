'''
def myGen():
  print("生成器被执行...")
  yield 1
  yield 2
myG = myGen()
print(next(myG))
'''

def libs():
  a = 0
  b = 1
  while True:
    a,b = b,a+b
    yield a
for each in libs():
  if each > 100:
    break
  print(each,end=' ')