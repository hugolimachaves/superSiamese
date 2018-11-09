import numpy as np

def conservative_similarity(list_zFeat):
	similarity = np.mean(list_zFeat[:list_zFeat.shape[0]/2], axis=0)

	print(similarity)
	print(similarity.shape)


zFeat = np.ones((1,6,6,256))
list_zFeat = np.zeros((1,6,6,256))


print(list_zFeat.shape)
list_zFeat = np.concatenate((list_zFeat, np.array(zFeat)))
zFeat = 2 * np.array(zFeat)
print(list_zFeat.shape)
list_zFeat = np.concatenate((list_zFeat, np.array(zFeat)))
zFeat = 3 * np.array(zFeat)
print(list_zFeat.shape)
list_zFeat = np.concatenate((list_zFeat, np.array(zFeat)))
zFeat = 4 * np.array(zFeat)
print(type(list_zFeat))
list_zFeat = np.concatenate((list_zFeat, np.array(zFeat)))
zFeat = 5 * np.array(zFeat)
print(type(list_zFeat))
list_zFeat = np.concatenate((list_zFeat, np.array(zFeat)))
zFeat = 6 * np.array(zFeat)
print(type(list_zFeat))


conservative_similarity(list_zFeat)
'''
'''
lista1 = []
lista2 = []

lista2.append(lista1)

print(lista2)

for i in lista2:
	print(i)
'''
'''
SIZE_DESCRIPTOR = 32

def getDescriptor():
	descriptor = []
	#TODO Estamos colocando apenas um place holder. A funcao depende da analise do tracker siameseFC no python
	for _ in range(SIZE_DESCRIPTOR):
		descriptor.append(float(np.random.randn()))
	
	return descriptor
'''
'''
descriptor = getDescriptor()
print(len(descriptor))
print(descriptor)
'''

'''
descriptor = np.random.randn(1,32)
print(descriptor.shape)
print(descriptor)

a = np.asarray(descriptor)
lista = []
for _ in range(3):
	lista.append(a)

print(lista)

lista = np.asarray(lista)
print(lista)
'''

'''
a.append(np.asarray(descriptor))
a.append(np.asarray(descriptor))
print(a)
a = np.asarray(a)

print(a)
print(type(a))
'''
'''
bb_list = []
bb_pos = []

for i in range(32):
	bb_pos.append(i)

	if(i%4==0 and i is not 0):
		bb_pos.append(i)
		bb_list.append(bb_pos)
		bb_pos = []
		print('')
		
	print('bb_list: '+str(bb_list))
print('bb_list: '+str(bb_list))
'''