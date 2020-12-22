def myFirstFunction(name):
  '函数定义是过程中的name是叫形参'
  print('我是实参name='+name)
print(myFirstFunction.__doc__)


def saySome(name,words):
  print(name + '->' +words)
saySome(words="喜欢python",name="我")


def text(*params):
  print('参数的长度是：',len(params))
  print('第二个参数是：',params[1])
text(1,'zhang',3.14,52)


def text2(*params,exp):
  print('参数的长度是：',len(params))
  print('第二个参数是：',params[1])
text2(1,'zhang',3.14,52,exp="exp")


def back():
  return [1,'zhang',3.14]
print(back())


count = 5
def myFun():
  global count
  count = 10
  print(count)
myFun()
print(count)


def fun1():
  print('fun1.........')
  def fun2():
    print('fun2......')
  fun2()
fun1()


def Funx(x):
  def Funy(y):
    return x * y
  return Funy
print(Funx(8)(5))


def Fun1():
  x = 5
  def Fun2():
    nonlocal x
    x *= x
    return x
  return Fun2()
print(Fun1())