try:
  int('abc')
  sum = 1 + '1'
  f = open('文件.txt')
  print(f.read())  
# except OSError as reason:
#   print('文件出错啦-.-\n错误的原因是：' + str(reason))
# except TypeError as reason:
#   print('类型出错啦-.-\n错误的原因是：' + str(reason))
except (OSError,TypeError):
  print('os,type错误')
except:
  print("出错啦！")
finally:
  f.close()