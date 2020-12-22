import random
'''
secret = random.randint(1,10)
temp = input("输入一个数字：")
guess = int(temp)
if guess == secret:
  print("你好聪明啊！")
else: 
  if guess > 8:
    print("大了")
  else: 
    print("小了")
print("游戏结束")
'''
score = int(input('请输入一个分数:'))
if 90 <= score <= 100:
  print('A')
elif 80 <= score < 90:
  print('B')
elif 70 <= score < 80:
  print('C')
else: 
  print('D')

x,y = 4,5
if x < y:
  small = x
else:
  small = y
print(small)
