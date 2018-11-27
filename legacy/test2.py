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

DEBUG_PRINT_ARRAY = True
MOSTRAR_OBJ_MODEL = True
DIM_DESCRIPTOR = 256
ONE_DIMENSION = 1
ATOMIC_SIZE = 88
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
K = 0.04
MAX_OBJECT_MODEL_ONE_DIM = 10
OBJECT_MODEL_DISPLAY_SIZE_ONE_DIM = 50
TOTAL_PIXEL_DISPLAY_OBJECT_MODEL_ONE_DIM = MAX_OBJECT_MODEL_ONE_DIM * OBJECT_MODEL_DISPLAY_SIZE_ONE_DIM


tf.set_random_seed(1)
#os.environ['PYTHONHASHSEED'] = '0'
#rn.seed(12345)
#np.random.seed(42)


CAMINHO_TESTE  = '/home/hugo/Documents/Mestrado/codigoSiameseTLD/siameseTLD'
NOME_VIDEO = 'godfather'
YML_FILE_NAME = 'parameters.yml'
CAMINHO_EXEMPLO_VOT2015 = '/home/hugo/Documents/Mestrado/vot2015/' + NOME_VIDEO

#CAMINHO_EXEMPLO_DATASET_TLD = '/home/hugo/Documents/Mestrado/codigoRastreador/dataset/exemplo/01-Light_video00001'
PARAMETERS_PATH = os.path.join(CAMINHO_EXEMPLO_VOT2015,YML_FILE_NAME)
print('caminho do yml:',PARAMETERS_PATH)


class Generation:
	
	def __init__(self,opts,siamiseNetWorkLocal):
		self.minimumSiameseNetPlaceHolder = tf.placeholder(tf.float32, [ONE_DIMENSION, opts['minimumSize'], opts['minimumSize'], RGB])
		tf.convert_to_tensor(False, dtype='bool', name='is_training')
		isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
		self.zMinimumPreTrained =siamiseNetWorkLocal.buildExemplarSubNetwork(self.minimumSiameseNetPlaceHolder,opts,isTrainingOp)
		self.tensorFlowSession = tf.Session()
		tf.initialize_all_variables().run(session=self.tensorFlowSession)

	#passando Bounding Box no formato Y,X,W,H, retornando left, top, right, botton
	def get_image_cropped(self, img, bb): # imagemCurrentFrame PIL
		left	= round(bb[0] - (bb[2]/2))
		top	    = round(bb[1] - (bb[3])/2)
		right   = round(bb[2] + left)
		bottom  = round(bb[3] + top)
		
		img = Image.fromarray(img,'RGB')
		img.show()
		cropped = img.crop([left,top,right,bottom])
		cropped.show()
		
		return cropped

	def getDescriptor(self,bb,imageSource): # zMinimumFeatures = sess.run(zMinimumPreTrained, feed_dict={minimumSiameseNetPlaceHolder: zCropMinimum})
		imImageSource = self.get_image_cropped(imageSource,bb)
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

	def convertYXWH2TLBR(self,yxwh):

		ltrb =[]
		ltrb.append(yxwh[0] - round(yxwh[3]/2) )
		ltrb.append(yxwh[1] - round(yxwh[2]/2) )
		ltrb.append(yxwh[0] + round(yxwh[3]/2) )
		ltrb.append(yxwh[1] + round(yxwh[2]/2) )
		return ltrb


	def crop_image(self, img, bb): # imagemCurrentFrame PIL
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

		return i, j


	def _get_frame_and_bb(self,BB_e_frame):

		frameNumber = BB_e_frame.pop(-1)
		return frameNumber, BB_e_frame


	def _get_modelo_shrinked_image(self,BB,imagem): # retorna a imagem shrinked referente ao frame enviado
		
		
		# 'imagem' com o shape okay, nao e (0,0,3)
		bb_ltrb = self.convertYXWH2TLBR(BB)
		image_cropped = self.crop_image(imagem,bb_ltrb) # entra np.array --> sai PIL image;
		#image_cropped.show()
		print('Bounding  box que provera o corte: ', BB)
		print('tamaho da imagem recortada: ', image_cropped.size)
		image_cropped =  np.array(image_cropped.getdata(),np.uint8).reshape(image_cropped.size[1],image_cropped.size[0],RGB)
		print('o shape de image Cropped e: ', image_cropped.shape)
		# abaixo esta dando imagem com dimensoes 0,0,3
		resized_image = cv2.resize(image_cropped,(self._size_subwindow_per_dimension,self._size_subwindow_per_dimension ),interpolation=cv2.INTER_CUBIC)
		cv2.imshow('imagem cv', resized_image)
		return resized_image


	def _adicionar_modelo_na_posicao(self,shrinked_image,i,j):

		#atribuicao
		print('O shape da imagem na rotina para adicionar modelos e: ',shrinked_image.shape)
		
		print('i inferior :', i, ' i superior :', i+1, 'j inferior: ', j , 'j superior: ', j+1)
		self._imagemModels[i*self._size_subwindow_per_dimension:(i+1)*self._size_subwindow_per_dimension, j*self._size_subwindow_per_dimension:(j+1)*self._size_subwindow_per_dimension] = shrinked_image


	def refreshObjectModel(self,objectModelList):

		if len(objectModelList) == self._number_models:

			pass

		else:

			for numero_do_modelo in range(self._number_models, len(objectModelList)):
				
				numero_do_frame, BB = self._get_frame_and_bb(objectModelList[numero_do_modelo])
				i,j = self._planarFromLinear(numero_do_modelo)
				shrinked = self._get_modelo_shrinked_image(BB, self._listFrames[numero_do_frame-1]) # porque o frame comeca do numero 1 e nao do numero 0. 
				self._adicionar_modelo_na_posicao(shrinked,i,j)

		# atualiza no numero o contador de modelos da classe
		self._number_models = len(objectModelList)
		self.imshow()



def convertSimilatiry(siameseDistance):
	return np.exp(- K * siameseDistance) # retorna a distancia no TLD
	#return 1.0 / (siameseDistance + 1.0) # retorna a distancia no TLD
	
def getLength(element): # verifica o tamanho total de elementos em uma estrutura de dados de dimensoes arbitrarias
	if isinstance(element, list):
		return sum(([getLength(i) for i in element]))
	return 1


def distCandidatesToTheModel(deep_features_candidates, isPositive=True):
	# passa  as deep Features dos candidatos para o presente frame conjuntamente
	# com o modelo positivo(default) ou negativo
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


	return np.sqrt(dist)

def getOpts(opts):
	print("config opts...")
	opts['numScale'] = 3
	opts['scaleStep'] = 1.0375
	opts['scalePenalty'] = 0.9745
	opts['scaleLr'] = 0.59
	opts['responseUp'] = 16
	opts['windowing'] = 'cosine'
	opts['wInfluence'] = 0.176
	opts['exemplarSize'] = 127
	opts['instanceSize'] = 255
	opts['scoreSize'] = 17
	opts['totalStride'] = 8
	opts['contextAmount'] = 0.5
	opts['trainWeightDecay'] = 5e-04
	opts['stddev'] = 0.03
	opts['subMean'] = False
	opts['minimumSize'] = ATOMIC_SIZE
	opts['video'] = NOME_VIDEO
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



generalDescriptor = DeepDescription()

def main(_):

	#[320,180,280,260], trump1
	#[320,200,230,260], trump2

	img = cv2.imread('car91.jpg',cv2.IMREAD_COLOR )#.astype(np.uint8)
	print('img shape: ', img.shape)
	print('tipo: ', type(img))
	
	aux = np.array(img[:,:,0])
	img[:,:,0] = img[:,:,2]
	img[:,:,2] = aux
	
	opts = configParams()
	opts = getOpts(opts)
	#add
	minimumSiameseNetPlaceHolder = tf.placeholder(tf.float32, [1, opts['minimumSize'], opts['minimumSize'], 3])
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
	saver.restore(sess, opts['modelName'])
	zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)
	zMinimumPreTrained =sn.buildExemplarSubNetwork(minimumSiameseNetPlaceHolder,opts,isTrainingOp)
	generated = Generation(opts,sn)

	descritores = []
	offset = 0
	for i in range(-1,2):
		descritores.append(generated.getDescriptor([340+offset,280+offset,160,100],img))
		offset +=50
	diff =  np.array(descritores[0]) -np.array(descritores[2])
	print('diferenca de descritores: ', diff)
	print('similaridade entre descritores: ', detSimilarity(descritores[0],descritores[2]))
	print(len(descritores))




if __name__=='__main__':
	tf.app.run()
