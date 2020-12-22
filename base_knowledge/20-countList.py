class CountList:
  def __init__(self,*args):
    self.values = [x for x in args]
    self.count = {}.fromkeys(range(len(self.values)),0)
  def __len__(self):
    return len(self.values)
  def __getitem__(self,key):
    self.count[key] +=1
    return self.values[key]
c1 = CountList(1,2,3,4,5)
c2 = CountList(2,4,5,6,7)
c1[1]
c2[1]
c1[1] + c2[1]
print(c1.count)