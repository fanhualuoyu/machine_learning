g = lambda x : 2*x +1
print(g(2))

h = lambda x,y : x+y
print(h(1,2))


temp = range(10)
print(list(filter(lambda x: x%2,temp)))

print(list(map(lambda x : x*2,temp)))