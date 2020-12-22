'''
class New_int(int):
  def __add__(self,other):
    return int.__sub__(self,other)
  def __sub__(self, other):
    return int.__add__(self,other)
a = New_int(3)
b = New_int(4)
print(a + b)

class Nint(int):
  def __radd__(self, value):
    return int.__sub__(self,value)
a = Nint(5)
b = Nint(3)
print(a+b)
print(1+b)
'''
import time

class MyTimer():
  def __init__(self):
    self.unit = ['年','月','日','时','分','秒']
    self.prompt = "未开始计时..."
    self.lasted = []
    self.begin = 0
    self.end = 0

  def __str__(self):
    return self.prompt
  __repr__ = __str__

  def __add__(self,other):
    prompt = "总共运行了"
    result = []
    for index in range(6):
      result.append(self.lasted[index] + other.lasted[index])
      if result[index]:
        prompt += (str(result[index]) + self.unit[index])
    return prompt

  #开始计时
  def start(self):
    self.begin = time.localtime()
    self.prompt = "提示：请先调用stop()停止计时"
    print("计时开始...")

  #停止计时
  def stop(self):
    if not self.begin:
      print("提示：请先调用start()进行计时")
    else:
      self.end = time.localtime()
      self._calc()
      print("计时结束...")

  #内部方法，计算运行时间  
  def _calc(self):
    self.lasted = []
    self.prompt = "总共运行了:"
    for index in range(6):
      self.lasted.append(self.end[index] - self.begin[index])
      if self.lasted:
        self.prompt += (str(self.lasted[index])) + self.unit[index]
    self.begin = 0
    self.end = 0

t1 = MyTimer()
t1.start()
t1.stop()

