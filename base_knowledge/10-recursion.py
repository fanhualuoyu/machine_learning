def factorial(n):
  if(n==1):
    return 1
  else:
    return n * factorial(n-1)
n = int(input("请输入一个数字："))
result = factorial(n)
print("%d 的阶乘是 %d" % (n,result))