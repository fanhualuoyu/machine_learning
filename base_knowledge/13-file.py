def saveFile(boy,girl,count):
  file_name_boy = 'boy_' + str(count) + '.txt'
  file_name_girl = 'girl_' + str(count) + '.txt'    
  boy_file = open(file_name_boy,'w')
  girl_file = open(file_name_girl,'w')
  boy_file.writelines(boy)
  girl_file.writelines(girl)
  boy_file.close()
  girl_file.close()

f = open('record.txt',encoding="UTF-8")

boy = []
girl = []
count = 1
for each_line in f:
  if(each_line[:6] != '======'):
    #这里进行字符串分割操作
    (role,line_spoken) = each_line.split('：',1)
    if role == '张':
      boy.append(line_spoken)
    else:
      girl.append(line_spoken)
  else:
    #文件的分别保存操作
    saveFile(boy,girl,count)
    boy = []
    girl = []
    count += 1
saveFile(boy,girl,count)
f.close

