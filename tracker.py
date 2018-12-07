#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import json
import socket
import localParameters as lp

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
FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK = 5 # infinito ==  original
SIAMESE_STRIDE = 8
SIAMESE_DESCRIPTOR_DIMENSION = 256
NUMBER_OF_EXEMPLAR_DESCRIPTOR = 6
AMPLITUDE_DESLOCAMENTO = 0 # define a amplitude da realizacao da media de templantes no espaco - 0 == original
FRAMES_COM_MEDIA_ESPACIAL = [POSICAO_PRIMEIRO_FRAME] # lista com o frames onde a media espacial sera realizada - [] ==  original


tf.set_random_seed(1) #os.environ['PYTHONHASHSEED'] = '0' #rn.seed(12345) #np.random.seed(42)



def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-v","--video"		,default=None, help = "nome da pasta do video")
	parser.add_argument("-n","--nomeSaida"	,default=None, help = "nome do arquivo de saida")
	parser.add_argument("-c","--caminho"	,default=None, help = "caminho ABSOLUTO para o dataset")
	parser.add_argument("-p","--parametro"	,default=None, help = "parametro a ser setado para esse tracker")
	return parser.parse_args()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_gaussian_values(size):
    s = 1
    g = gaussian(size-1, 0, s)

    while True:
        if((1/size - g) <= 0.0001):
            break
        elif(1/size >= g):
            s -= 0.001
            g = gaussian(size-1, 0, s)
        else:
            s += 0.001
            g = gaussian(size-1, 0, s)

    print('s = ',s)
    print('g = ',g)

    g_list = []
    for i in range(n):
        g_list.append(gaussian(i, 0, s))

    return g_list

class SuperTemplate:
	'''
	retorna um superTemplate formato tf.constant
	'''
	templateList = []

	def __init__(self):
		pass

	def conservativeTemplate(self):
		return tf.constant( sum(self.templateList[:round(len(self.templateList)/2)])/round(len(self.templateList)/2) , dtype=tf.float32) 

	def oneShotTemplate(self):
		return tf.constant( self.templateList[0] , dtype=tf.float32) 
	
	def innovativeTemplate(self,nLast):
		return tf.constant( sum(self.templateList[-nLast:])/len(self.templateList[-nLast:]) , dtype=tf.float32) 
		
	def progressiveTemplate(self):
		return tf.constant( sum(self.templateList[-round(len(self.templateList)/2):])/round(len(self.templateList)/2) , dtype=tf.float32) 

	def nShotTemplate(self,nshots):
		return tf.constant( sum(self.templateList[:nshots])/len(self.templateList[:nshots]) , dtype=tf.float32) 

	def cummulativeTemplateMod(self,zFeat,frame,template):
		print('frame: ',frame)
		p1 = (1/(frame+1))
		p2 = (frame/(frame+1))
		return (tf.constant(zFeat , dtype=tf.float32) * p1 + template * p2)

	def cummulativeTemplate(self):
		return tf.constant( sum(self.templateList)/len(self.templateList) , dtype=tf.float32)

	# mode : Forma da distribuicao
	#	- 0: Maior peso no inicio da lista (Frame mais Antigo)
	#	- 1: Maior peso no meio da lista
	#	- 2: Maior peso no final da lista (Frame mais Recente)
	def mediaMovelGaussiana(self, size=15, mode=1):
		if(mode == 0 or mode == 2):
			g = get_gaussian_values(size)
			if(mode == 2):
				g = g[::-1] # Get Reverse List

		else: # (mode == 1)
			half = int(1 + (size/2))
			g = get_gaussian_values(half)

			if(size % 2 == 1):
			    g_a = g[::-1] # Get Reverse List
			    g = g[1:]
			else:
			    g = g[1:]
			    g_a = g[::-1] # Get Reverse List

			g = g_a+g

		aux = []
		for pos, i in enumerate(self.templateList[len(self.templateList)-size:]):
			aux.append(i*g[pos])

		return tf.constant( sum(aux)/g.sum , dtype=tf.float32) 

	def addInstance(self,instance):
		template = np.array(instance)
		self.templateList.append(template)



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

def get_next_frame(imgFiles, frame):
	return cv2.imread(imgFiles[frame]).astype(np.float32)

def get_list_img_files(vpath):
	imgs = []
	imgFiles = [imgFile for imgFile in glob.glob(os.path.join(vpath, "*.jpg"))]
	for imgFile in imgFiles:
		if imgFile.find('00000000.jpg') >= 0:
			imgFiles.remove(imgFile)

	imgFiles.sort()

	return imgFiles

def loadVideoInfo(basePath, video):
	videoPath = os.path.join(basePath, video)
	groundTruthFile = os.path.join(basePath, video, 'groundtruth.txt')

	groundTruth = open(groundTruthFile, 'r')
	reader = groundTruth.readline()
	region = [float(i) for i in reader.strip().split(",")]

	cx, cy, w, h = getAxisAlignedBB(region)
	pos = [cy, cx]
	targetSz = [h, w]

	imgFiles = get_list_img_files(videoPath)

	return imgFiles, np.array(pos), np.array(targetSz)

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
	with tf.Session() as sess:
		template = sess.run(template)

	#template = tf.Session().run(template) # casting para np array

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
	#construção da criacao do objeto com media espacial
	
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

	show = lp.getInJson('tracker','show')
	opts = configParams()
	opts = getOpts(opts)
	#add
	caminhoDataset = caminho_do_dataset
	caminhoVideo = os.path.join(caminhoDataset,nome_do_video)
	caminhoLog =  os.path.join(caminhoVideo,'__log__')
	nome_log = nome_do_arquivo_de_saida
	parametro = int(parametro)
	FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK  = int(parametro)


	#REDE 1
	exemplarOp = tf.placeholder(tf.float32, [1, opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOp = tf.placeholder(tf.float32, [opts['numScale'], opts['instanceSize'], opts['instanceSize'], 3])
	exemplarOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
	instanceOpBak = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
	isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')
	sn = SiameseNet()
	scoreOpBak = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts, isTraining=False)
	saver = tf.train.Saver()
	#writer = tf.summary.FileWriter(opts['summaryFile'])
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
	#writer2 = tf.summary.FileWriter(opts['summaryFile'])
	sess2 = tf.Session()
	saver2.restore(sess2, opts['modelName'])
	zFeatOp2 = sn2.buildExemplarSubNetwork(exemplarOp2, opts, isTrainingOp2)

	#imgs, targetPosition, targetSize = loadVideoInfo(caminhoDataset, nome_do_video)
	imgFiles, targetPosition, targetSize = loadVideoInfo(caminhoDataset, nome_do_video)


	nImgs = len(imgFiles)
	#imgs_pil =  [Image.fromarray(np.uint8(img)) for img in imgs]

	im = get_next_frame(imgFiles, POSICAO_PRIMEIRO_FRAME)

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
	oldTemplate = tf.constant(zFeat, dtype=tf.float32)
	scoreOp = sn.buildInferenceNetwork(instanceOp, template, opts, isTrainingOp)
	#writer.add_graph(sess.graph)

	#REDE2
	zCrop_original = np.array(zCrop)
	zFeat_original = sess2.run(zFeatOp2, feed_dict={exemplarOp2: zCrop_original})
	zFeat_original = np.transpose(zFeat_original, [1, 2, 3, 0])
	template_original = tf.constant(zFeat_original, dtype=tf.float32)
	#template = np.array(template_original)
	template = tf.identity(template_original)
	oldTemplate = tf.identity(template_original) 
	template_acumulado = np.array(template)
	scoreOp_original = sn.buildInferenceNetwork(instanceOp, template_original, opts, isTrainingOp)
	#writer2.add_graph(sess2.graph)

	teste1 =  tf.constant(zFeat,dtype=tf.float32)
	teste2 =  tf.Session().run(teste1)
	teste3  = tf.constant(teste2,dtype=tf.float32)

	
	#assert 2 == 1

	tic = time.time()
	ltrb = []

	superDescritor = SuperTemplate()
	superDescritor.addInstance(np.array(zFeat))

	print('zfeat:' , zFeat[0,0,-10,0] )
	

	for frame in range(POSICAO_PRIMEIRO_FRAME, nImgs):
		
		im = get_next_frame(imgFiles, frame)

		print(('frame ' + str(frame+1) + ' / ' + str(nImgs)).center(80,'*'))

		if frame > POSICAO_PRIMEIRO_FRAME:

			zCrop, _ = getSubWinTracking(im, targetPosition, (opts['exemplarSize'], opts['exemplarSize']), (np.around(sz), np.around(sz)), avgChans)
			zCrop = np.expand_dims(zCrop, axis=0)
			zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop})
			zFeat = np.transpose(zFeat, [1, 2, 3, 0])
			zFeat.reshape(1,NUMBER_OF_EXEMPLAR_DESCRIPTOR,NUMBER_OF_EXEMPLAR_DESCRIPTOR,SIAMESE_DESCRIPTOR_DIMENSION)


			if frame < FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK:
				superDescritor.addInstance(np.array(zFeat_original))
			else:
				superDescritor.addInstance(np.array(zFeat))


			if(im.shape[-1] == 1): # se a imagem for em escala de cinza
				tmp = np.zeros([im.shape[0], im.shape[1], 3], dtype=np.float32)
				tmp[:, :, 0] = tmp[:, :, 1] = tmp[:, :, 2] = np.squeeze(im)
				im = tmp
			scaledInstance = sx * scales
			scaledTarget = np.array([targetSize * scale for scale in scales])
			xCrops = makeScalePyramid(im, targetPosition, scaledInstance, opts['instanceSize'], avgChans, None, opts)
			
			'''
			if frame < FRAMES_TO_ACUMULATE_BEFORE_FEEDBACK:
				template = template_original
			else:
				template = superDescritor.mediaMovelGaussiana(size=20, mode=0)
			'''

			template = superDescritor.cummulativeTemplate()

			
			with tf.Session() as sess1:
				template = sess1.run(template)
				template = tf.constant(template , dtype=tf.float32)


			#template_espacial = spatialTemplate (targetPosition,im, opts, sz, avgChans,sess,zFeatOp,exemplarOp,FRAMES_COM_MEDIA_ESPACIAL,amplitude = 0, cumulative = False, adaptative = False )
			#template = superDescritor.cummulativeTemplate()
			#template = superDescritor.progressiveTemplate()
			#template = superDescritor.nShotTemplate(3)
			#
			
			#template = template_original

			#filtro adaptativo logo abaixo:
			#template = filtroAdaptativo(template,zFeat,parametro)
			#~filtro adaptativo
			
			scoreOp = sn.buildInferenceNetwork(instanceOp, template, opts, isTrainingOp)
			score = sess.run(scoreOp, feed_dict={instanceOp: xCrops})
			#sio.savemat('score.mat', {'score': score})
			newTargetPosition, newScale = trackerEval(score, round(sx), targetPosition, window, opts)
			targetPosition = newTargetPosition
			sx = max(minSx, min(maxSx, (1-opts['scaleLr'])*sx+opts['scaleLr']*scaledInstance[newScale]))
			targetSize = (1-opts['scaleLr'])*targetSize+opts['scaleLr']*scaledTarget[newScale]

		else:
			pass

		rectPosition = targetPosition-targetSize/2.
		tl = tuple(np.round(rectPosition).astype(int)[::-1])
		br = tuple(np.round(rectPosition+targetSize).astype(int)[::-1])
		if show: # plot only if it is in a desktop that allows you to watch the video
			imDraw = im.astype(np.uint8)
			cv2.putText(imDraw,str(frame+1)+'/'+str(nImgs),(0,25),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),2)
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

if __name__== '__main__':
	
	args = _get_Args()
	listCustom = [args.video,args.nomeSaida,args.caminho,args.parametro]
	listDefault = [lp.getInJson('process','video_teste'), lp.getInJson('process','nome_saida'), lp.getInJson('tracker','datasetPath'), lp.getInJson('process','parametro')]
	listaArgs = [argumentoDefault if argumentoCustom == None else argumentoCustom  for argumentoCustom, argumentoDefault in zip(listCustom,listDefault)] # colcoar argumentos default caso nao sejam passados argumentos costumizaveis
	_main(listaArgs[0],listaArgs[1],listaArgs[2],listaArgs[3])
