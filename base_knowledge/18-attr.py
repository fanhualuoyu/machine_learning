'''
class C:  
  #定义当该类的属性被访问时的行文
  def __getattribute__(self, name):
    print("getattribute")
    return super().__getattribute__(name)
  #定义当用户试图获取一个不存在的属性时的行为
  def __getattr__(self,name):
    print("getattr")
  #定义当一个属性被设置时的行为
  def __setattr__(self, name, value):
    print("setattr")
    super().__setattr__(name,value)
  #定义一个属性被删除时的行为
  def __delattr__(self, name):
    print("delattr")
    return super().__delattr__(name)

c = C()
c.x = 1
del c.x
'''

class Rectangle:
  def __init__(self,width = 0,height = 0):
    self.width = width
    self.height = height
  def __setattr__(self, name, value):
    if name == "square":
      self.width = value
      self.height = value
    else:
      super().__setattr__(name,value)
  def getArea(self):
      return self.width * self.height

r = Rectangle(4,5)
print(r.getArea())
r.square = 10
print(r.getArea())