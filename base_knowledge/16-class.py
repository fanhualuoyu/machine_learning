class Turtle:
  #属性
  color = 'green'
  weight = 10
  legs = 4
  shell = True
  mouth = '大嘴'

  #方法
  def climb(self):
    print("climb..........")
  def run(self):
    print("run............")
  def bite(self):
    print("bite...........")
  def eat(self):
    print("eat............")
  def sleep(self):
    print("sleep..........")

t = Turtle()
t.bite()


class Person:
  __name = "python"
  def getName(self):
    return self.__name
p = Person()
print(p.getName())


class Parent:
  def hello(self):
    print("parent........")
class Child(Parent):
  def hello(self):
    print("Child......")
  pass
p = Parent()
p.hello()
c = Child()
c.hello()


class a:
  pass
class b(a):
  pass
print(issubclass(b,a))
b1 = b()
print(isinstance(b1,b))

class c:
  def __init__(self,size=10):
    self.size = size
  def getSize(self):
    return self.size
  def setSize(self,value):
    self.size = value
  def delSize():
    del self.size
  x = property(getSize,setSize,delSize)
c1 = c()
print(c1.x)
c1.x = 18
print(c1.x)