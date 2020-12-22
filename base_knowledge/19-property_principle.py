'''
class MyDecriptor:
  #用于访问属性，它返回属性的值
  def __get__(self,instance,owner):
    print("get...",self,instance,owner)
  #将在属性分配操作中调用，不返回任何内容
  def __set__(self,instance,value):
    print("set...",self,instance,value)
  #控制删除操作，不返回任何内容
  def __delete__(self,instance):
    print("deleting...",self,instance)
class Test:
  x = MyDecriptor()

test = Test()
test.x
del test.x
'''

class MyProperty:
  def __init__(self,fget = None,fset = None,fdel = None):
    self.fget = fget
    self.fset = fset
    self.fdel = fdel
  #self 指的是MyProperty类的实例对象x
  #instance 指的是C类的实例对象c
  #owner 指的就是C类本身
  def __get__(self,instance,owner):
    return self.fget(instance)
  def __set__(self,instance,value):
    self.fset(instance,value)
  def __delete__(self,instance):
    self.fdel(instance)
class C:
  def __init__(self):
    self._x = None
  def getX(self):
    return self._x
  def setX(self,value):
    self._x = value
  def delX(self):
    del self._x
  x = MyProperty(getX,setX,delX)
c = C()
c.x = "x-main"
print(c._x)