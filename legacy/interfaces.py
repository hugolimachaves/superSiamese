import numpy as np
from ctypes import *
from sklearn.neighbors import KNeighborsClassifier
import os

shared_library = CDLL('TLD/bin/Debug/libTLD.so')

SIZE_ARRAY = 32
LAST_ADDED = -1
SIZE_DESCRIPTOR = 256

HULL_SIZE = 4
WINDOW_SIZE = 40 # Example bb_list
OBJECT_MODEL_SIZE = 400 # Example
	
ARRAY_SIZE = 500 # Example

a = 0

bad_windows  = []
good_windows = []
good_windows_hull   = []
positive_obj_model	= []
negative_obj_model	= []
feature_pos_obj_model = []
feature_neg_obj_model = []

# good_windows_hull[ N ][ 5 ]
#   - N: Numero de bb no frame
#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

# positive_obj_model[ P ][ N ][ 5 ]
#   - P: Numero de frames que retornaram bb ate o momento
#   - N: Numero de bb do frame no p-esimo frame
#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

# negative_obj_model[ P ][ N ][ 5 ]
#   - P: Numero de frames que retornaram bb ate o momento
#   - N: Numero de bb do frame no p-esimo frame
#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

# good_windows[ P ][ N ][ 5 ]
#   - P: Numero de frames que retornaram bb ate o momento
#   - N: Numero de bb do frame no p-esimo frame
#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

def convertSimilatiry(siameseDistance):
	return 1 / (siameseDistance + 1) # retorna a distancia no TLD
	
def getLength(element): # verifica o tamanho total de elementos em uma estrutura de dados de dimensoes arbitrarias
	if isinstance(element, list):
		return sum(([getLength(i) for i in element]))
	return 1

def getDescriptor(bb):
	descriptor = []
	#TODO Estamos colocando apenas um place holder. A funcao depende da analise do tracker siameseFC no python
	#for _ in range(SIZE_DESCRIPTOR):
	
	descriptor = np.random.randn(1,SIZE_DESCRIPTOR)

	return descriptor

# passa  as deep Features dos candidatos para o presente frame conjuntamente
# com o modelo positivo(default) ou negativo
def distCandidatesToTheModel(deep_features_candidates, isPositive=True):
	#Usa os seguintes parametros globais:  feature_pos_obj_model, feature_neg_obj_model  
	features = []


	if isPositive: # modelo positivo do object model
		positiveLabel = [1 for i in feature_pos_obj_model]
		labels = np.asarray(positiveLabel)
		features = np.asarray(feature_pos_obj_model)
	
	else: # modelo negativo do object model
		negativeLabel = [0 for i in feature_neg_obj_model] 
		labels = np.asarray(negativeLabel)
		features = np.asarray(feature_neg_obj_model)
	
	distances = []
	positions = []

	#print('features.size ', features.size)
	if (features.size is not 0):
		knn_1 = KNeighborsClassifier(n_neighbors=1)

		if(False):
			print('features len printando:',len(features[LAST_ADDED]))
			print('Dimensao de features: ', features.shape ,' Dimensao de label', labels.shape)
			
			print("labels".center(70,'*'))
			print(labels)

			print("features".center(70,'*'))
			print(features)
			#print('type of features: ', type(features), ' shape of features: ', features.reshape((features.shape[-1]) ))
	
		listFeatures = [bb for frame in features for bb in frame]
		knn_1.fit(listFeatures, labels)
		
		for candidate in deep_features_candidates: # pega a menor distancia para cada candidato na lista deep_features_candidate
			list_candidate = np.asarray(candidate)
			dist,position = knn_1.kneighbors(list_candidate, n_neighbors=1, return_distance=True)
			distances.append(dist[0][0])
			positions.append(position)
			# example: neigh.kneighbors([[1., 1., 1.]])
			# pode das errado porque a documentacao mostra um array de array

	return distances # retorna a menor distancia em relacao ao modelo, eh uma lista pois sao varios candidatos e tambem  a posicao no vetor

	#return distances # retorna a menor distancia em relacao ao modelo, eh uma lista pois sao varios candidatos e tambem  a posicao no vetor

# passo uma lista de bb dos candidatos e uma bb do tracker (lista de lista)
def detSimilarity(candidates, bb_tracker, is_neg_empty=False, is_pos_empty=False, is_candidates_empty=False, is_bb_tracker_empty=False):
	# candidates:
	# sera utilizado - (sao globais): positive_obj_model, negative_obj_model

	positive_distances_candidates = []
	negative_distances_candidates = []
	positive_distances_tracker = []
	negative_distances_tracker = []

	if(not is_neg_empty):
		bb_list_model_last = [BB for BB in negative_obj_model[LAST_ADDED]]

		feature_aux = []
		for bb in bb_list_model_last:
			descriptor = getDescriptor(bb)
			feature_neg_obj_model.append(descriptor) # o tamanho e equivalente ao numero de descritores negativos
			feature_aux.append(descriptor)

		'''
		print('neg ', negative_obj_model[LAST_ADDED])
		print('len feat ', len(feature_aux),'len neg mod ',len(negative_obj_model[LAST_ADDED]))
		'''

		assert(len(feature_aux) == len(negative_obj_model[LAST_ADDED])), 'o tamanho do object Model difere do tamanho dos descritores'

	if(not is_pos_empty):
		bb_list_model_last = [BB for BB in positive_obj_model[LAST_ADDED]]

		feature_aux = []
		for bb in bb_list_model_last:	
			descriptor = getDescriptor(bb)
			feature_pos_obj_model.append(descriptor)  # o tamanho e equivalente ao numero de descritores positivos
			feature_aux.append(descriptor)

		'''	
		print('feature_pos_obj_model: ', feature_pos_obj_model, 'tamanho: ',feature_pos_obj_model[0].shape)
		print('neg ', positive_obj_model[LAST_ADDED])
		print('asd ', bb_list_model)
		print('len feat ', getLength(feature_aux),'len pos mod ',getLength(positive_obj_model))
		print('\npositive_obj_model: ', positive_obj_model)
		'''

		assert(len(feature_aux) == len(positive_obj_model[LAST_ADDED])), 'o tamanho do object Model difere do tamanho dos descritores'

	if(not is_candidates_empty):
		deep_features_candidates = []
		for bb in candidates[LAST_ADDED]:
			#TODO fazer o descritor
			descriptor = getDescriptor(bb)
			'''
			print('descritor'.center(50,'*'))
			print('O descritor provisorio e: ', descriptor)
			print('~descritor'.center(50,'*'))
			'''
			deep_features_candidates.append(descriptor)
		deep_features_candidates = np.asarray(deep_features_candidates)
		#print('Info deep_features_candidates: ',  deep_features_candidates, '\n tipo: ', type(deep_features_candidates) )

		positive_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=True)
		negative_distances_candidates = distCandidatesToTheModel(deep_features_candidates, isPositive=False)

	if(not is_bb_tracker_empty):
		feature_tracker = []
		for bb in bb_tracker[LAST_ADDED]:
			descriptor = getDescriptor(bb)
			feature_tracker.append(descriptor)
		feature_tracker =  np.asarray(feature_tracker)

		positive_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=True)
		negative_distances_tracker = distCandidatesToTheModel(feature_tracker, isPositive=False)

	'''
	print('positive_distances_candidates: ',positive_distances_candidates)
	print('negative_distances_candidates: ',negative_distances_candidates)
	print('positive_distances_tracker: ',positive_distances_tracker)
	print('negative_distances_tracker: ',negative_distances_tracker)
	'''

	positive_siam_sim_cand = [convertSimilatiry(distancia) for distancia in positive_distances_candidates]
	negative_siam_sim_cand = [convertSimilatiry(distancia) for distancia in negative_distances_candidates]
	positive_siam_sim_bb_tracker = [convertSimilatiry(distancia) for distancia in positive_distances_tracker]
	negative_siam_sim_bb_tracker = [convertSimilatiry(distancia) for distancia in negative_distances_tracker]

	'''
	print('positive_siam_sim_cand: ',positive_siam_sim_cand)
	print('negative_siam_sim_cand: ',negative_siam_sim_cand)
	print('positive_siam_sim_bb_tracker: ',positive_siam_sim_bb_tracker)
	print('negative_siam_sim_bb_tracker: ',negative_siam_sim_bb_tracker)
	'''

	return positive_siam_sim_cand, negative_siam_sim_cand, positive_siam_sim_bb_tracker, negative_siam_sim_bb_tracker

def read_data(array, array_size, frame):
	bb_list = []
	is_empty = True
	print('array_size: ',array_size)
	if(array_size is not 0):
		bb_pos = []
		for i in range(array_size):
			bb_pos.append(array[i])

			if(i%4==0 and i is not 0):
				bb_pos.append(frame)
				bb_list.append(bb_pos)
				bb_pos = []

		bb_pos.append(frame)
		bb_list.append(bb_pos)
		bb_pos = []

		is_empty = False

	return bb_list, is_empty

#'frame' se refere ao numero do frame que esta sendo processado no codigo .py
def init_interface(frame=2): 
	'''
	codigo de execucao do c/c++ aqui!
	 
	Parametros: (frame ou void)
 
	Retorno do codigo:
	1) lista com as posicoes extraidas, 'lista1'. pode ser uma lista de estruturas de 5 valores,
		(4 referentes a localizacao do frame, e um indicando se o modelo eh positivo-1- ou negativo-0)
		Caso precise retornar um numero fixo de valores, entre em contato comigo. A principio, se precisar retornar
		um vetor de tamanho fixo, adote um vetor de 100 estruturas de 5 posicoes. Onde nao ha nada preenchido, coloque
		valores negativos, como: -1
	2)Frame ao qual foi processada as informacoes, que alimentara a variavel: retorno_frame.
	Vaviavel necessaria para garantir o processamento do mesmo frame.
	'''

	#print("Caminho atual e:",)
	parameters_path = os.getcwd() + "/dataset/exemplo/01-Light_video00001/parameters.yml"
	parameters_path = parameters_path.encode('utf-8')

	retorno_frame = c_int() # numero do frame atual

	size_negative = c_int(0) # tamanho do vetor array objectModel negativo
	size_positive = c_int(0) # tamanho do vetor array objectModel positivo
	size_good_windows = c_int(0) # tamanho do vetor array good windows
	size_good_windows_hull = c_int(0) # tamanho do vetor array good_windows_hull (que e sempre 4)

	array_good_windows          = [-1] * WINDOW_SIZE # placeholders
	array_good_windows_hull     = [-1] * HULL_SIZE # placeholders
	array_object_model_negative = [-1] * OBJECT_MODEL_SIZE # placeholders
	array_object_model_positive = [-1] * OBJECT_MODEL_SIZE # placeholders

	# fazendo a alocacao dos vetores, (*array_good_windows) --> posicao de memoria do vetor
	array_good_windows 			= (c_float * WINDOW_SIZE) (*array_good_windows) 
	array_good_windows_hull     = (c_float * 4) (*array_good_windows_hull)
	array_object_model_negative = (c_float * OBJECT_MODEL_SIZE) (*array_object_model_negative) 
	array_object_model_positive = (c_float * OBJECT_MODEL_SIZE) (*array_object_model_positive) 

	shared_library.initializer_TLD(parameters_path, byref(retorno_frame), 
								   array_object_model_positive, byref(size_positive), 
								   array_object_model_negative, byref(size_negative),
								   array_good_windows, byref(size_good_windows),
								   array_good_windows_hull,  byref(size_good_windows_hull))
	
	print('\nFrame de entrada: '+ str(frame)+ ' Frame de retorno: ' + str(retorno_frame.value) + '\n')
	assert (frame == retorno_frame.value), "Conflito nos frames"

	bb_list, is_neg_empty = read_data(array_object_model_negative, size_negative.value, frame)
	if(not is_neg_empty):
		negative_obj_model.append(bb_list)

	bb_list, is_pos_empty = read_data(array_object_model_positive, size_positive.value, frame)
	if(not is_pos_empty):
		positive_obj_model.append(bb_list)

	bb_list, is_good_empty = read_data(array_good_windows, size_good_windows.value, frame)
	if(not is_good_empty):
		good_windows.append(bb_list)

	bb_list, is_good_hull_empty = read_data(array_good_windows_hull, size_good_windows_hull.value, frame)
	if(not is_good_hull_empty):
		good_windows.append(bb_list)

	'''
	print('\n\n')
	if(size_good_windows_hull.value is not 0):
		print(' Os valores do good windows hull '.center(70,'*'))
		bb_pos = []
		for i in range(size_good_windows_hull.value):
			bb_pos.append(array_good_windows_hull[i])	
			print(str(array_good_windows_hull[i])+' ',end='')

		print()
		bb_pos.append(frame)

		good_windows_hull.append(bb_pos)
		is_good_hull_empty = False
	'''

def TLD(frame):
	retorno_frame = c_int()

	size_candidates = c_int()
	size_positive   = c_int()
	size_negative   = c_int()
	size_bb_tracker = c_int()

	array_bb_candidates		 = [-1] * ARRAY_SIZE
	array_object_model_positive = [-1] * ARRAY_SIZE
	array_object_model_negative = [-1] * ARRAY_SIZE

	array_bb_candidates		 = (c_float * ARRAY_SIZE) (*array_bb_candidates)
	array_object_model_positive = (c_float * ARRAY_SIZE) (*array_object_model_positive)
	array_object_model_negative = (c_float * ARRAY_SIZE) (*array_object_model_negative)

	array_bb_tracker = [-1] * 4
	array_bb_tracker = (c_float * 4) (*array_bb_tracker)

	shared_library.TLD_function_1(byref(retorno_frame), array_bb_candidates, byref(size_candidates),
									array_object_model_positive, byref(size_positive), 
									array_object_model_negative, byref(size_negative),
									array_bb_tracker, byref(size_bb_tracker))
	
	print('\nFrame de entrada: '+ str(frame)+ ' Frame de retorno: ' + str(retorno_frame.value))
	assert (frame == retorno_frame.value), "Conflito nos frames"

	candidates = []
	bb_tracker = []

	is_candidates_empty = True
	is_bb_tracker_empty = True

	bb_list, is_neg_empty = read_data(array_object_model_negative, size_negative.value, frame)
	if(not is_neg_empty):
		negative_obj_model.append(bb_list)

	bb_list, is_pos_empty = read_data(array_object_model_positive, size_positive.value, frame)
	if(not is_pos_empty):
		positive_obj_model.append(bb_list)

	bb_list, is_candidates_empty = read_data(array_bb_candidates, size_candidates.value, frame)
	if(not is_candidates_empty):
		candidates.append(bb_list)

	bb_list, is_bb_tracker_empty = read_data(array_bb_tracker, size_bb_tracker.value, frame)
	if(not is_bb_tracker_empty):
		bb_tracker.append(bb_list)

	'''
	# BB de referencia do tracker
	if(size_bb_tracker.value is not 0):
		print('\n\n')
		print(' Os valores das bb tracker '.center(70,'*'))
		bb_list = []
		bb_pos = []
		for i in range(size_bb_tracker.value):
			bb_pos.append(array_bb_tracker[i])				
			print(str(array_bb_tracker[i])+' ',end='')

		bb_pos.append(frame)

		bb_tracker.append(bb_pos)
		is_bb_tracker_empty = False
	print('\n\n')
	'''

	# candidates[ N ][ 5 ]
	#   - N: Numero de bb no frame
	#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

	# bb_tracker[ 1 ][ 5 ]
	#   - 1: Para ser considerado uma lista no algoritmos
	#   - 5: (Centro_X, Centro_Y, Width, Height, Frame_Number)

	
	positive_distances_candidates = []
	negative_distances_candidates = []
	positive_distances_tracker = []
	negative_distances_tracker = []



	# passa uma lista de bb dos candidatos retornado pelo TLD e passa uma bb retornado pelo Tracker no TLD
	positive_distances_candidates, negative_distances_candidates, positive_distances_tracker, negative_distances_tracker = detSimilarity(candidates, bb_tracker, is_neg_empty, is_pos_empty, is_candidates_empty, is_bb_tracker_empty)

	'''
	print('positive_distances_candidates'.center(70,'*'),positive_distances_candidates)
	print('negative_distances_candidates'.center(70,'*'),negative_distances_candidates)
	print('positive_distances_tracker'.center(70,'*'),positive_distances_tracker)
	print('negative_distances_tracker'.center(70,'*'),negative_distances_tracker)
	'''
	retorno_frame = c_int()

	size_good_windows      = c_int(0) # tamanho do vetor array good windows
	size_good_windows_hull = c_int(0) # tamanho do vetor array good_windows_hull (que e sempre 4)

	similaridade_positiva_candidates = [-1] * ARRAY_SIZE
	similaridade_negativa_candidates = [-1] * ARRAY_SIZE
	similaridade_positiva_bb_tracker = [-1] * ARRAY_SIZE
	similaridade_negativa_bb_tracker = [-1] * ARRAY_SIZE

	array_good_windows      = [-1] * ARRAY_SIZE
	array_good_windows_hull = [-1] * ARRAY_SIZE

	similaridade_positiva_candidates = (c_float * ARRAY_SIZE) (*similaridade_positiva_candidates)
	similaridade_negativa_candidates = (c_float * ARRAY_SIZE) (*similaridade_negativa_candidates)
	similaridade_positiva_bb_tracker = (c_float * ARRAY_SIZE) (*similaridade_positiva_bb_tracker)
	similaridade_negativa_bb_tracker = (c_float * ARRAY_SIZE) (*similaridade_negativa_bb_tracker)

	array_good_windows      = (c_float * ARRAY_SIZE) (*array_good_windows)
	array_good_windows_hull = (c_float * ARRAY_SIZE) (*array_good_windows_hull)

	shared_library.TLD_function_2(similaridade_positiva_candidates, similaridade_negativa_candidates,
								  similaridade_positiva_bb_tracker, similaridade_negativa_bb_tracker,
								  array_good_windows, byref(size_good_windows),
								  array_good_windows_hull, byref(array_good_windows_hull))

init_interface(1)

for i in range(2,354):
	TLD(i)
 
#'deepDescriptor' eh o descritor que sera passado para o codigo C. eh um descritor de 128/256 floats.
def interface2(deepDescriptor,frame):

	'''
	codigo de execucao do c/c++ aqui!
	 
	Parametros( deepDescriptor e (frame ou void)
 
	1)retorna 'objectModel' que eh uma lista de posicoes do dos modelos de objetos detectados nesse, e somente nesse, frame,
	Onde a estrutura tem 5 elementos, 4 para posicoes e 1 para indicar se esse posicao eh positiva ou negativa. Retornne tambehm
	o numero do frame processado, para verificacao do assert.
	Caso precise retorna uma lista de tamanho fixo (um vetor), siga as recomendacoes do item 1) do comentario para interface1.
	2)Frame ao qual foi processada as informacoes, que alimentara a variavel: 'retornoFrame'.
	Vaviavel necessaria para garantir o processamento do mesmo frame.
	'''
	assert (frame==retornoFrame), "Conflito nos frames"
	return objectModel

