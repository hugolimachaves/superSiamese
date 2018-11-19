# Suofei ZHANG, 2017.
import os
import math
import tensorflow as tf
import numpy as np
import glob
import cv2
import scipy.io as sio
import time
from PIL import Image
import utils
from siamese_net import SiameseNet
from parameters import configParams
from ctypes import *
from sklearn.neighbors import KNeighborsClassifier
import random as rn
import argparse



DEBUG_TRACKER = False
DEBUG_PRINT_ARRAY = False
MOSTRAR_OBJ_MODEL = False
DEBUG_3 = False
DIM_DESCRIPTOR = 256
ONE_DIMENSION = 1
ATOMIC_SIZE = 87
SIZE_ARRAY = 32
LAST_ADDED = -1
SIZE_DESCRIPTOR = 256
HULL_SIZE = 4
WINDOW_SIZE = 40 # Example bb_list
OBJECT_MODEL_SIZE = 400 # Example
ARRAY_SIZE = 500 # Example
RGB = 3
POSICAO_PRIMEIRO_FRAME = 0
POSICAO_SEGUNDO_FRAME = 1
PRIMEIRO_FRAME = 1
SEGUNDO_FRAME = 2
ULTIMO_FRAME = -1
K = 0.01#0.04
MAX_OBJECT_MODEL_ONE_DIM = 10
OBJECT_MODEL_DISPLAY_SIZE_ONE_DIM = 50
TOTAL_PIXEL_DISPLAY_OBJECT_MODEL_ONE_DIM = MAX_OBJECT_MODEL_ONE_DIM * OBJECT_MODEL_DISPLAY_SIZE_ONE_DIM
CAMINHO_DATASET = '/home/hugo/Documents/Mestrado/vot2015'

FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK = 5 # infinito ==  original

IN_A_SERVER = True
SIAMESE_STRIDE = 8
SIAMESE_DESCRIPTOR_DIMENSION = 256
NUMBER_OF_EXEMPLAR_DESCRIPTOR = 6
AMPLITUDE_DESLOCAMENTO = 0 # define a amplitude da realizacao da media de templantes no espaco - 0 == original
FRAMES_COM_MEDIA_ESPACIAL = [POSICAO_PRIMEIRO_FRAME] # lista com o frames onde a media espacial sera realizada - [] ==  original
MI = 0.01#0.1 # parametro do filtro adaptativo - 0 == original.

tf.set_random_seed(1) #os.environ['PYTHONHASHSEED'] = '0' #rn.seed(12345) #np.random.seed(42)

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("video", help= "nome da pasta do video")
	parser.add_argument("nome", help = "nome do arquivo de saida")
	parser.add_argument("caminho", help ="caminho ABSOLUTO para o dataset")
	parser.add_argument("parametro", help = "parametro a ser setado para esse tracker")
	return parser.parse_args()


class Generation:

	def __init__(self,opts,siamiseNetWorkLocal):
		self.minimumSiameseNetPlaceHolder = tf.placeholder(tf.float32, [ONE_DIMENSION, opts['minimumSize'], opts['minimumSize'], RGB])
		tf.convert_to_tensor(False, dtype='bool', name='is_training')
		isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
		self.zMinimumPreTrained =siamiseNetWorkLocal.buildExemplarSubNetwork(self.minimumSiameseNetPlaceHolder,opts,isTrainingOp)
		self.tensorFlowSession = tf.Session()
		tf.initialize_all_variables().run(session=self.tensorFlowSession)


	def  extend(self,bb,margin):
		bb_aux = []
		bb_aux.append(round(bb[0] - bb[2]/2 - abs((bb[2]))*margin))
		bb_aux.append(round(bb[1] - bb[3]/2 - abs((bb[3]))*margin))
		bb_aux.append(round(bb[2] + bb[2]/2 + abs((bb[2]))*margin))
		bb_aux.append(round(bb[3] + bb[3]/2 + abs((bb[3]))*margin))
		return bb_aux


	#passando Bounding Box no formato Y,X,W,H, retornando left, top, right, botton
	def get_image_cropped(self, img, bb): # imagemCurrentFrame PIL
		left	= round(bb[0] - (bb[2]/2))
		top	    = round(bb[1] - (bb[3]/2))
		right   = round(bb[2] + left)
		bottom  = round(bb[3] + top)

		cropped = img.crop([left,top,right,bottom])

		return cropped

	def convertYXWH2LTRB(self,yxwh):

		ltrb =[]
		ltrb.append(yxwh[0] - round(yxwh[3]/2) )
		ltrb.append(yxwh[1] - round(yxwh[2]/2) )
		ltrb.append(yxwh[0] + round(yxwh[3]/2) )
		ltrb.append(yxwh[1] + round(yxwh[2]/2) )
		return ltrb

	def get_image_cropped2(self, img, bb, margem):

		bb = self.convertYXWH2LTRB(bb[:4])


		bb = self.extend(bb,margem)

		img_np = np.array(img)

		mean_color = [np.mean(img_np[:,:,0]), np.mean(img_np[:,:,1]), np.mean(img_np[:,:,2])]
		shape_original = img_np.shape[:2]
		boundary = [0, 0, img_np.shape[1] , img_np.shape[0]]
		margin = []

		for i in range(len(boundary)):
			if(i < 2):
				margin.append(min(bb[i] - boundary[i], 0))
			else:
				margin.append(min(boundary[i] - bb[i], 0))

		img_cropped = img.crop(bb)
		img_np = np.array(img_cropped)

		for i in range(RGB):

			if(margin[0] is not 0):
				img_np[0:-margin[0], :, i] = mean_color[i]
			if(margin[1] is not 0):
				img_np[:, 0:-margin[1], i] = mean_color[i]
			if(margin[2] is not 0):
				img_np[img_np.shape[0]+margin[3]:img_np.shape[0], :, i] = mean_color[i]
			if(margin[3] is not 0):
				img_np[:, img_np.shape[1]+margin[2]:img_np.shape[1], i] = mean_color[i]

		img_cropped = Image.fromarray(img_np)

		return img_cropped

	'''
	def get_image_cropped_ltrb(self, img, bb): # imagemCurrentFrame PIL
		left	= round(bb[0] - (bb[2]/2))
		top	    = round(bb[1] - (bb[3])/2)
		right   = round(bb[2] + left)
		bottom  = round(bb[3] + top)

		cropped = img.crop([left,top,right,bottom])
		return cropped
	'''

	def getDescriptor(self,bb,imageSource): # zMinimumFeatures = sess.run(zMinimumPreTrained, feed_dict={minimumSiameseNetPlaceHolder: zCropMinimum})
		imImageSource = self.get_image_cropped2(imageSource,bb, 0.35) ##MARGEM
		#imImageSource.show()
		#imageSource.show()
		#input('iaperta alguma coisa')
		neoImageSource = imImageSource.resize((ATOMIC_SIZE,ATOMIC_SIZE))
		npImageSource = np.array(neoImageSource)
		npImageSource = npImageSource.reshape(1,npImageSource.shape[0],npImageSource.shape[1],3)
		zMinimumFeatures = self.tensorFlowSession.run(self.zMinimumPreTrained, feed_dict={self.minimumSiameseNetPlaceHolder: npImageSource})

		return zMinimumFeatures

class DeepDescription:
	positive_obj_model_bb		 = []
	negative_obj_model_bb 		 = []
	good_windows_bb 	 		 = []
	good_windows_hull_bb 		 = []
	tracker_bb					 = []
	__candidates_bb 			 = []

	positive_obj_model_features  = []
	negative_obj_model_features  = []
	good_windows_features 		 = []
	good_windows_hull_features   = []
	tracker_features			 = []
	__candidates_features 		 = []

	#TODO Colocar privado
	positive_distances_candidates  = []
	negative_distances_candidates  = []
	positive_similarity_candidates = []
	negative_similarity_candidates = []

	#TODO Colocar privado
	positive_distances_tracker_candidate  = []
	negative_distances_tracker_candidate  = []
	positive_similarity_tracker_candidate = []
	negative_similarity_tracker_candidate = []

	__currentFrame = 0

	def __init__(self):
		self.__currentFrame = 0

	def setCandidates(self,candBB,candFeat,currentFrame):
		self.__currentFrame = currentFrame

		self.__candidates_bb = []
		self.__candidates_bb = candBB

		self.__candidates_features = []
		self.__candidates_features = candFeat

	def getCandidates(self,currentFrame):
		if (currentFrame is self.__currentFrame):
			return self.__candidates_bb, self.__candidates_features

		self.__currentFrame = currentFrame
		return [], []

class Visualization:


	def __init__(self, n_subWindows, size_subWindow,title, listFrames):

		self._n_subwindows_per_dimension = n_subWindows
		self._size_subwindow_per_dimension =  size_subWindow
		self._n_pixels_display_one_dim = self._n_subwindows_per_dimension* self._size_subwindow_per_dimension
		self._number_models = 0
		self._imagemModels =  np.zeros((self._n_pixels_display_one_dim , self._n_pixels_display_one_dim, RGB ), dtype=np.uint8)
		self._titulo = title
		self._listFrames = listFrames

	def convertYXWH2LTRB(self,yxwh):

		ltrb =[]
		ltrb.append(yxwh[0] - round(yxwh[3]/2) )
		ltrb.append(yxwh[1] - round(yxwh[2]/2) )
		ltrb.append(yxwh[0] + round(yxwh[3]/2) )
		ltrb.append(yxwh[1] + round(yxwh[2]/2) )
		return ltrb

	'''
	def addObjectModelVisualization(objecModel,img,): # formato para opencv: ltrb

		objectModel = list(objectModel)
		if len(objectModeL== 5):
			objectModel = objectModel(:-1)
		objectModel =  =convertYXWH2LTRB(objectModel)
	'''
	def crop_image(self, img, bb, percentMargin = 0): # imagemCurrentFrame PIL
		img_pil = Image.fromarray(np.uint8(img))
		cropped = img_pil.crop(bb)
		return cropped


	def imshow(self):

		cv2.imshow(self._titulo,self._imagemModels)


	def destroyWindow(self):

		try:
			cv2.destroyWindow(self._titulo)
		except:
			print('ERRO: Nao foi possivel destruir a janela')

	def _planarFromLinear(self,linear):
		if linear == 0:
			i = 0
			j = 0
		else: # verificar isso aqui

			j = linear%self._n_subwindows_per_dimension
			i = int(linear/self._n_subwindows_per_dimension)%self._n_subwindows_per_dimension

			''' errado
			linear = self._n_subwindows_per_dimension**2%linear
			j = linear%self._n_subwindows_per_dimension
			i = int(linear/self._n_subwindows_per_dimension)
			'''
		return i, j


	def _get_frame_and_bb(self,BB_e_frame):

		frameNumber = BB_e_frame.pop(-1)
		return frameNumber, BB_e_frame


	def _get_modelo_shrinked_image(self,BB,imagem): # retorna a imagem shrinked referente ao frame enviado

		# 'imagem' com o shape okay, nao e (0,0,3)
		bb_ltrb = self.convertYXWH2LTRB(BB)
		image_cropped = self.crop_image(imagem,bb_ltrb) # entra np.array --> sai PIL image;
		image_cropped =  np.array(image_cropped.getdata(),np.uint8).reshape(image_cropped.size[1],image_cropped.size[0],RGB)
		# abaixo esta dando imagem com dimensoes 0,0,3
		resized_image = cv2.resize(image_cropped,(self._size_subwindow_per_dimension,self._size_subwindow_per_dimension ),interpolation=cv2.INTER_CUBIC)
		#cv2.imshow('imagem cv', resized_image)
		return resized_image


	def _adicionar_modelo_na_posicao(self,shrinked_image,i,j):

		#atribuicao
		if(DEBUG_3):
			print('O shape da imagem na rotina para adicionar modelos e: ',shrinked_image.shape)
			print('i inferior :', i, ' i superior :', i+1, 'j inferior: ', j , 'j superior: ', j+1)
		self._imagemModels[i*self._size_subwindow_per_dimension:(i+1)*self._size_subwindow_per_dimension, j*self._size_subwindow_per_dimension:(j+1)*self._size_subwindow_per_dimension] = shrinked_image


	def refreshObjectModel(self,objectModelList):

		if len(objectModelList) == self._number_models:

			pass

		else:

			for numero_do_modelo in range(self._number_models, len(objectModelList)):
				#print('Entrando na atulaizacao de modelo')
				numero_do_frame, BB = self._get_frame_and_bb(objectModelList[numero_do_modelo])
				i,j = self._planarFromLinear(numero_do_modelo)
				shrinked = self._get_modelo_shrinked_image(BB, self._listFrames[numero_do_frame-1]) # porque o frame comeca do numero 1 e nao do numero 0. 7
				#cv2.imshow('teste',shrinked)
				self._adicionar_modelo_na_posicao(shrinked,i,j)

		# atualiza no numero o contador de modelos da classe
		self._number_models = len(objectModelList)
		self.imshow()

'''
imshow
refreshModel

'''



generalDescriptor = DeepDescription()

def convertSimilatiry(siameseDistance):
	return np.exp(- K * siameseDistance) # retorna a distancia no TLD
	#return 1.0 / (siameseDistance + 1.0) # retorna a distancia no TLD

def getLength(element): # verifica o tamanho total de elementos em uma estrutura de dados de dimensoes arbitrarias
	if isinstance(element, list):
		return sum(([getLength(i) for i in element]))
	return 1

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

	if (features.size is not 0):
		knn_1 = KNeighborsClassifier(n_neighbors=1)
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

# passo duas features e calcula a distancia euclidiana entre elas
def detSimilarity(feature_a, feature_b):
	dist = 0
	if len(feature_a.shape) > 2:
		feature_a = feature_a.reshape(-1)
	if len(feature_b.shape) > 2:
		feature_b = feature_b.reshape(-1)
	for a, b in zip(feature_a, feature_b):
		dist += (a - b) ** 2

	'''
	#Comparacao de duas features positivas para identificar o melhor K no convertSimilatiry
	feature_1 = generalDescriptor.good_windows_features[-1].reshape(-1)
	feature_2 = generalDescriptor.good_windows_features[-2].reshape(-1)

	dist = 0
	for a, b in zip(feature_1, feature_2):
		#print('(a - b) ** 2: ', (a - b) ** 2,'\n')
		dist += (a - b) ** 2

	print('\n\ndist: ',np.sqrt(dist))
	print('convertSimilatiry: ',convertSimilatiry(float(np.sqrt(dist))))
	#print('feature_1: ', feature_1.reshape(-1))
	#print('feature_2: ', feature_2.reshape(-1))
	'''

	'''
	print('feature_a: ', feature_a.reshape(-1))
	print('feature_b: ', feature_b.reshape(-1))
	print('feature_a - feature_b: ', feature_a.reshape(-1) - feature_b.reshape(-1))
	print('dist: ', float(np.sqrt(dist)))
	'''

	return np.sqrt(dist)

def read_data(array, array_size, frame, name=0):
	bb_list = []
	is_empty = True

	if DEBUG_PRINT_ARRAY and name is not 0:
		if(name == 1):
			print('\n\tNegativo ', end='')
		if(name == 2):
			print('\n\tPositivo ', end='')
		if(name == 3):
			print('\n\tCandidatos ', end='')
		if(name == 4):
			print('\n\tBoundding Box do tracker ', end='')

		print('array: ',end='')

	if(array_size != 0):
		bb_pos = []
		for i in range(array_size):
			if((i%4==0) and (i != 0)):
				bb_pos.append(frame)
				bb_list.append(bb_pos)
				bb_pos = []

			bb_pos.append(array[i])

			if (DEBUG_PRINT_ARRAY) and (name != 0):
				if i%4 == 0:
					print('[', end='')
					print(array[i],', ', end='')
				elif i%4 == 3:
					print(array[i],end='')
					print('] ', end='')
				else:
					print(array[i],', ',end='')

		bb_pos.append(frame)
		bb_list.append(bb_pos)
		bb_pos = []
		is_empty = False

	return bb_list, is_empty

def getOpts(opts):
	print("config opts...")
	opts['numScale'] = 3
	opts['scaleStep'] = 1.0375
	opts['scalePenalty'] = 0.9745
	opts['scaleLr'] = 0.59
	opts['responseUp'] = 16
	opts['windowing'] =  'cosine' #''uniform'#'
	opts['wInfluence'] = 0.176
	opts['exemplarSize'] = 127
	opts['instanceSize'] = 255
	opts['scoreSize'] = 17
	opts['totalStride'] = 8
	opts['contextAmount'] = 0.5
	opts['trainWeightDecay'] = 5e-04
	opts['stddev'] = 0.03
	opts['subMean'] = False
	opts['minimumSize'] = 'invalido'
	opts['video'] = 'invalido'
	opts['modelPath'] = './models/'
	opts['modelName'] = opts['modelPath']+"model_tf.ckpt"
	opts['summaryFile'] = './data_track/'+opts['video']+'_20170518'

	return opts

def getAxisAlignedBB(region):
	region = np.array(region)
	nv = region.size
	assert (nv == 8 or nv == 4)

	if nv == 8:
		xs = region[0 : : 2] #comeca do zero e incrementa de 2 em 2
		ys = region[1 : : 2] #comeca do um e incrementa de 2 em 2
		cx = np.mean(xs)
		cy = np.mean(ys)
		x1 = min(xs)
		x2 = max(xs)
		y1 = min(ys)
		y2 = max(ys)
		A1 = np.linalg.norm(np.array(region[0:2])-np.array(region[2:4]))*np.linalg.norm(np.array(region[2:4])-np.array(region[4:6]))
		A2 = (x2-x1)*(y2-y1)
		s = np.sqrt(A1/A2)
		w = s*(x2-x1)+1
		h = s*(y2-y1)+1
	else:
		x = region[0]
		y = region[1]
		w = region[2]
		h = region[3]
		cx = x+w/2
		cy = y+h/2

	return cx-1, cy-1, w, h

def frameGenerator(vpath):
	imgs = []
	imgFiles = [imgFile for imgFile in glob.glob(os.path.join(vpath, "*.jpg"))]
	for imgFile in imgFiles:
		if imgFile.find('00000000.jpg') >= 0:
			imgFiles.remove(imgFile)

	imgFiles.sort()

	for imgFile in imgFiles:
		# imgs.append(mpimg.imread(imgFile).astype(np.float32))
		# imgs.append(np.array(Image.open(imgFile)).astype(np.float32))
		img = cv2.imread(imgFile).astype(np.float32)
		imgs.append(img)

	return imgs

def loadVideoInfo(basePath, video):
	videoPath = os.path.join(basePath, video)
	groundTruthFile = os.path.join(basePath, video, 'groundtruth.txt')

	groundTruth = open(groundTruthFile, 'r')
	reader = groundTruth.readline()
	region = [float(i) for i in reader.strip().split(",")]

	cx, cy, w, h = getAxisAlignedBB(region)
	pos = [cy, cx]
	targetSz = [h, w]

	imgs = frameGenerator(videoPath)

	return imgs, np.array(pos), np.array(targetSz)

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
	if originalSz is None:
		originalSz = modelSz

	sz = originalSz
	im_sz = img.shape
	# make sure the size is not too small
	assert min(im_sz[:2]) > 2, "the size is too small"
	c = (np.array(sz) + 1) / 2
	# check out-of-bounds coordinates, and set them to black
	context_xmin = round(pos[1] - c[1])
	context_xmax = context_xmin + sz[1] - 1
	context_ymin = round(pos[0] - c[0])
	context_ymax = context_ymin + sz[0] - 1
	left_pad = max(0, int(-context_xmin))
	top_pad = max(0, int(-context_ymin))
	right_pad = max(0, int(context_xmax - im_sz[1] + 1))
	bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))
	context_xmin = int(context_xmin + left_pad)
	context_xmax = int(context_xmax + left_pad)
	context_ymin = int(context_ymin + top_pad)
	context_ymax = int(context_ymax + top_pad)
	if top_pad or left_pad or bottom_pad or right_pad:
		r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[0])
		g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[1])
		b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
				   constant_values=avgChans[2])
		r = np.expand_dims(r, 2)
		g = np.expand_dims(g, 2)
		b = np.expand_dims(b, 2)
		img = np.concatenate((r, g, b ), axis=2)
	im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
	if not np.array_equal(modelSz, originalSz):
		im_patch = cv2.resize(im_patch_original, modelSz)
	else:
		im_patch = im_patch_original

	return im_patch, im_patch_original

def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans, stats, p):
	'''
	computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
	and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.

	'''
	in_side_scaled = np.round(in_side_scaled)
	max_target_side = int(round(in_side_scaled[-1]))
	min_target_side = int(round(in_side_scaled[0]))
	beta = out_side / float(min_target_side)
	# size_in_search_area = beta * size_in_image
	# e.g. out_side = beta * min_target_side
	search_side = int(round(beta * max_target_side))
	search_region, _ = getSubWinTracking(im, targetPosition, (search_side, search_side),
											  (max_target_side, max_target_side), avgChans)
	if p['subMean']:
		pass
	assert round(beta * min_target_side) == int(out_side)
	tmp_list = []
	tmp_pos = ((search_side - 1) / 2., (search_side - 1) / 2.)
	for s in range(p['numScale']):
		target_side = round(beta * in_side_scaled[s])
		tmp_region, _ = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
											   avgChans)
		tmp_list.append(tmp_region)
	pyramid = np.stack(tmp_list)
	return pyramid

def trackerEval(score, sx, targetPosition, window, opts):
	# responseMaps = np.transpose(score[:, :, :, 0], [1, 2, 0])
	responseMaps = score[:, :, :, 0]
	upsz = opts['scoreSize']*opts['responseUp']
	# responseMapsUp = np.zeros([opts['scoreSize']*opts['responseUp'], opts['scoreSize']*opts['responseUp'], opts['numScale']])
	responseMapsUP = []

	if opts['numScale'] > 1:
		currentScaleID = int(opts['numScale']/2)
		bestScale = currentScaleID
		bestPeak = -float('Inf')
		for s in range(opts['numScale']):
			if opts['responseUp'] > 1:
				responseMapsUP.append(cv2.resize(responseMaps[s, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
			else:
				responseMapsUP.append(responseMaps[s, :, :])
			thisResponse = responseMapsUP[-1]
			if s != currentScaleID:
				thisResponse = thisResponse*opts['scalePenalty']
			thisPeak = np.max(thisResponse)
			if thisPeak > bestPeak:
				bestPeak = thisPeak
				bestScale = s
		responseMap = responseMapsUP[bestScale]
	else:
		responseMap = cv2.resize(responseMaps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
		bestScale = 0
	responseMap = responseMap - np.min(responseMap)
	responseMap = responseMap/np.sum(responseMap)
	responseMap = (1-opts['wInfluence'])*responseMap+opts['wInfluence']*window
	rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
	pCorr = np.array((rMax, cMax))
	dispInstanceFinal = pCorr-int(upsz/2)
	dispInstanceInput = dispInstanceFinal*opts['totalStride']/opts['responseUp']
	dispInstanceFrame = dispInstanceInput*sx/opts['instanceSize']
	newTargetPosition = targetPosition+dispInstanceFrame

	return newTargetPosition, bestScale

def getCumulativeTemplate(zFeat,frame,template):
	return (tf.constant(zFeat , dtype=tf.float32) * (1/(frame+1)) + template * (frame/(frame+1)))


def filtroAdaptativo(template,zFeat,mi):

	#filtro adaptativo
	y = np.zeros([NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR])
	e = np.zeros([NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR])
	d = np.zeros([NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR])
	template = tf.Session().run(template) # casting para np array

	for i in range(NUMBER_OF_EXEMPLAR_DESCRIPTOR):
		for j in range(NUMBER_OF_EXEMPLAR_DESCRIPTOR):
			
			template256 = np.array(template[i,j,:,0])
			zFeat256 = np.array(zFeat[i,j,:,0])
			template256 = np.reshape(template256,(256,1))
			zFeat256 = np.reshape(zFeat256,(256,1))
			y[i,j] = np.inner(np.reshape(template256,-1) , np.reshape(zFeat256,256))
			d[i,j] = np.inner(template[i,j,:,0],template[i,j,:,0]) 
			e[i,j] = d[i,j] - y[i,j]
			template[i,j,:,0] = template[i,j,:,0] - mi*zFeat[i,j,:,0]*e[i,j] 
	template = tf.constant(template , dtype=tf.float32) # casting para np array ser uma tf constant
	return template

def spatialTemplate(targetPosition,im, opts, sz, avgChans,sess,zFeatOp,exemplarOp,FRAMES_COM_MEDIA_ESPACIAL,amplitude = 0, cumulative = False, adaptative = False ):
	#construção da criaçao do objeto com media espacial
	
	#TODO - Isso aqui sera chamado fora da funcao. A ser Feito
	if frame in FRAMES_COM_MEDIA_ESPACIAL:
		amplitude = AMPLITUDE_DESLOCAMENTO
		assert 1==2 # definir a amplitude de deslocamento
		assert amplitude <= SIAMESE_STRIDE # nao faz sentido um deslocamento maior que esse, pois voce ira pegar "celulas de descricao" iguais - lembre-se que que o stride e 8	
	else:
		amplitude = 0

	spatial_cont = 0
	#quando a media espacial na for desejada, ambos lacos aninhados serao executarao apenas uma iteracao
	for desloc_x in range(-amplitude,amplitude+1):
		for desloc_y in range(-amplitude,amplitude+1):
			targetPosition[0] = targetPosition[0] + desloc_x
			targetPosition[1] = targetPosition[1] + desloc_y
			zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)
			zCrop = np.expand_dims(zCrop, axis=0)
			zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
			zFeat = np.transpose(zFeat, [1, 2, 3, 0])
			zFeat.reshape(1,NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR,SIAMESE_DESCRIPTOR_DIMENSION)
			template = tf.constant(zFeat , dtype=tf.float32) * (1/(spatial_cont+1)) + template*(spatial_cont/(spatial_cont+1))
			spatial_cont+=1 # caso seja feita uma media espacial, deve-se incrementar o contador para aumentar a variavel no inicio do loop
			#pegando o template cumulativo
	if (cumulative):
		template = getCumulativeTemplate(zFeat,frame,template) #TODO:> verifcar a condidional para a utilizacao do template  acumulativo
	#filtro adaptativo
	if (adaptative):
		template = filtroAdaptativo(template,zFeat,spatial_cont)

	return template



'''----------------------------------------main-----------------------------------------------------'''
def _main(nome_do_video,nome_do_arquivo_de_saida,caminho_do_dataset,parametro):

	opts = configParams()
	opts = getOpts(opts)
	#add
	caminhoDataset = caminho_do_dataset
	caminhoVideo = os.path.join(caminhoDataset,nome_do_video)
	caminhoLog =  os.path.join(caminhoVideo,'__log__')
	nome_log = nome_do_arquivo_de_saida
	mi = float(parametro)

	#REDE 1
	exemplarOp = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOp = tf.placeholder(tf.float32, [opts['numScale'], opts['instanceSize'], opts['instanceSize'], 3])
	exemplarOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
	isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
	sn = SiameseNet()
	scoreOpBak = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts, isTraining=False)
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(opts['summaryFile'])
	sess = tf.Session()
	sess2 = tf.Session()
	saver.restore(sess, opts['modelName'])
	saver.restore(sess2, opts['modelName'])
	zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)

	#REDE2
	exemplarOp2 = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOp2 = tf.placeholder(tf.float32, [opts['numScale'], opts['instanceSize'], opts['instanceSize'], 3])
	exemplarOpBak2 = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOpBak2 = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
	isTrainingOp2 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
	sn2 = SiameseNet()
	#	scoreOpBak2 = sn2.buildTrainNetwork(exemplarOpBak2, instanceOpBak2, opts, isTraining=False)
	saver2 = tf.train.Saver()
	writer2 = tf.summary.FileWriter(opts['summaryFile'])
	sess2 = tf.Session()
	saver2.restore(sess2, opts['modelName'])
	zFeatOp2 = sn2.buildExemplarSubNetwork(exemplarOp2, opts, isTrainingOp2)

	imgs, targetPosition, targetSize = loadVideoInfo(  caminhoDataset, nome_do_video )
	nImgs = len(imgs)
	imgs_pil =  [Image.fromarray(np.uint8(img)) for img in imgs]


	im = imgs[POSICAO_PRIMEIRO_FRAME]
	if(im.shape[-1] == 1):
		tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
		tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
		im = tmp

	avgChans = np.mean(im, axis=(0, 1))# [np.mean(np.mean(img[:, :, 0])), np.mean(np.mean(img[:, :, 1])), np.mean(np.mean(img[:, :, 2]))]
	wcz = targetSize[1]+opts['contextAmount']*np.sum(targetSize)
	hcz = targetSize[0]+opts['contextAmount']*np.sum(targetSize)
	sz = np.sqrt(wcz*hcz)
	scalez = opts['exemplarSize']/sz

	zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)
	zCrop2, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)

	if opts['subMean']:
		pass

	dSearch = (opts['instanceSize']-opts['exemplarSize'])/2
	pad = dSearch/scalez
	sx = sz+2*pad
	minSx = 0.2*sx
	maxSx = 5.0*sx
	winSz = opts['scoreSize']*opts['responseUp']
	if opts['windowing'] == 'cosine':

		hann = np.hanning(winSz).reshape(winSz, 1)
		window = hann.dot(hann.T)
	elif opts['windowing'] == 'uniform':
		window = np.ones((winSz, winSz), dtype=np.float32)

	window = window/np.sum(window)
	scales = np.array([opts['scaleStep'] ** i for i in range(int(np.ceil(opts['numScale']/2.0)-opts['numScale']), int(np.floor(opts['numScale']/2.0)+1))])

	#REDE1
	zCrop = np.expand_dims(zCrop, axis=0)
	zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
	zFeat = np.transpose(zFeat, [1, 2, 3, 0])
	template = tf.constant(zFeat, dtype=tf.float32)
	scoreOp = sn.buildInferenceNetwork(instanceOp, template, opts, isTrainingOp)
	writer.add_graph(sess.graph)

	#REDE2
	zCrop_original = np.array(zCrop)
	zFeat_original = sess2.run(zFeatOp2, feed_dict={exemplarOp2: zCrop_original})
	zFeat_original = np.transpose(zFeat_original, [1, 2, 3, 0])
	template_original = tf.constant(zFeat_original, dtype=tf.float32)
	template = np.array(template_original)
	template_acumulado = np.array(template)
	scoreOp_original = sn.buildInferenceNetwork(instanceOp, template_original, opts, isTrainingOp)
	writer2.add_graph(sess2.graph)

	tic = time.time()
	ltrb = []


	for frame in range(POSICAO_PRIMEIRO_FRAME, nImgs):

		im = imgs[frame]

		print(('frame ' + str(frame+1)).center(80,'*'))
		if frame > POSICAO_PRIMEIRO_FRAME:

			zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)
			zCrop = np.expand_dims(zCrop, axis=0)
			zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
			zFeat = np.transpose(zFeat, [1, 2, 3, 0])
			zFeat.reshape(1,NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR,SIAMESE_DESCRIPTOR_DIMENSION)
			
			#template = tf.constant(zFeat , dtype=tf.float32) * (1/(frame+1)) + template*(frame/(frame+1))
			
			if(im.shape[-1] == 1): # se a imagem for em escala de cinza
				tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
				tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
				im = tmp
			scaledInstance = sx * scales
			scaledTarget = np.array([targetSize * scale for scale in scales])
			xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None, opts)

			'''
			template  = getCumulativeTemplate(zFeat,frame,template)
			if frame < FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK:
				template = template_original
			'''
			
			#template_espacial = spatialTemplate (targetPosition,im, opts, sz, avgChans,sess,zFeatOp,exemplarOp,FRAMES_COM_MEDIA_ESPACIAL,amplitude = 0, cumulative = False, adaptative = False )
			
			if frame < 2:
				template = template_original

			print("shape de template antes de entrar na funcao do filtro adaptativo: ",template[0,1,:,0].shape )
			template = filtroAdaptativo(template,zFeat,mi)

			
			scoreOp = sn.buildInferenceNetwork(instanceOp, template, opts, isTrainingOp)
			score = sess.run(scoreOp, feed_dict={instanceOp: xCrops})
			sio.savemat('score.mat', {'score': score})
			newTargetPosition, newScale = trackerEval(score, round(sx), targetPosition, window, opts)
			targetPosition = newTargetPosition
			sx = max(minSx, min(maxSx, (1-opts['scaleLr'])*sx+opts['scaleLr']*scaledInstance[newScale]))
			targetSize = (1-opts['scaleLr'])*targetSize+opts['scaleLr']*scaledTarget[newScale]

		else:
			pass

		rectPosition = targetPosition-targetSize/2.
		tl = tuple(np.round(rectPosition).astype(int)[::-1])
		br = tuple(np.round(rectPosition+targetSize).astype(int)[::-1])
		if not IN_A_SERVER: # plot only if it is in a desktop that allows you to watch the video
			imDraw = im.astype(np.uint8)
			cv2.rectangle(imDraw, tl, br, (0, 255, 255), thickness=3)
			cv2.imshow("tracking - siamese", imDraw)
			cv2.waitKey(1)
		ltrb.append(list(tl) + list(br))
	with open( os.path.join(caminhoLog, nome_log ),'w') as file:
		linhas = []
		for i in ltrb:
			linha = ''
			for cont,coord in enumerate(i) :
				if cont == 3:
					linha = linha + str(coord) + '\n'
				else:
					linha = linha + str(coord) + ','
			linhas.append(linha)
		for i in linhas:
			file.write(i)
	print(time.time()-tic)
	return

if __name__=='__main__':

	args = _get_Args()
	video = args.video
	nome_do_arquivo_de_saida = args.nome
	caminho_do_dataset = args.caminho
	parametro = args.parametro
	_main(video,nome_do_arquivo_de_saida,caminho_do_dataset, parametro)
