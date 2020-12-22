import pickle
my_list = [123,3.14,'python',['another','list']]
pickle_file = open('my_list.pkl','wb')
pickle.dump(my_list,pickle_file)
pickle_file.close()

pickle_file = open('my_list.pkl','rb')
my_list2 = pickle.load(pickle_file)
print(my_list2)