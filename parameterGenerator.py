import os 
import math
import yxwh_generator as yxwh
import pandas as pd
import xlwt
import localParameter as lp



LTRB2 = 0
LTRB4 = 1
YXWH  = 2


def iou(box1, box2): 
	'''Implements intersection over union (IoU) between box1 and box2: box1 -- first box, list object coordinates (x1, y1, x2, y2) --box2 -- second box, list object coordinates (x1, y1, x2, y2)'''
	'''
	if len(box1) == 8:
		box1 = converterCoord(box1)
	if len(box2) == 8:
		box2 = converterCoord(box2)
	'''
	assert (len(box1) == 4) and (len(box2) == 4)

	# Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
	xi1 = max(box1[0],box2[0])
	yi1 = max(box1[1],box2[1])
	xi2 = min(box1[2],box2[2])
	yi2 = min(box1[3],box2[3])
	# Max between zero  and the difference is for the case where the intersection doesn't exist, ie, its empty and thus zero
	inter_area = max(xi2 - xi1, 0)*max(yi2 - yi1, 0)

	# Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	box1_area = (box1[2] - box1[0])*(box1[3] - box1[1])
	box2_area = (box2[2] - box2[0])*(box2[3] - box2[1])
	union_area = box1_area + box2_area - inter_area

	return inter_area/union_area

def convertYXWH2LTRB(yxwh):

	ltrb =[]
	ltrb.append(yxwh[0] - round(yxwh[2]/2) )
	ltrb.append(yxwh[1] - round(yxwh[3]/2) )
	ltrb.append(yxwh[0] + round(yxwh[2]/2) )
	ltrb.append(yxwh[1] + round(yxwh[3]/2) )
	return ltrb

def determineAndWriteIoU(root,gtFile,outputFile,logs,comparisonTitle):

	try:
		os.mkdir(logs)
	except:
		pass
	with open(os.path.join(root,gtFile),'r') as file:
		gt = [list(map(float, line.split(','))) for line in file.read().splitlines()]
	
	with open(os.path.join(root,outputFile), 'r') as file:
		predict = [list(map(float, line.split(','))) for line in file.read().splitlines()]
	
	with open(os.path.join(root,logs,comparisonTitle),'w') as file:		
		file.writelines([str(iou(box1,box2)) + '\n' for box1, box2 in zip(gt, predict)])



def determineIoU(gtList,outputList,root,entrada):

	entrada = str(entrada)
	with open(os.path.join(root,gtList),'r') as file:
		gt = [list(map(float, line.split(','))) for line in file.read().splitlines()]
	
	if int(entrada) ==  int(LTRB2):
		pass
	if int(entrada) == int(LTRB4):
		gt = [ yxwh.getAxisAlignedBB(linha) for linha in gt if len(linha)==8]# se for coordenadas poligonais, deve-se converter para ywwh.
		gt = [convertYXWH2LTRB(linha) for linha in gt]
	if int(entrada) == int(YXWH):
		gt = [convertYXWH2LTRB(linha) for linha in gt]
	with open(os.path.join(root,outputList), 'r') as file:
		predict = [list(map(float, line.split(','))) for line in file.read().splitlines()]
	iouResult = [(iou(box1,box2)) for box1, box2 in zip(gt, predict)]
	return iouResult


def datasetIterateIoU(caminhoBase, gtFile,outputFile,rootOutput,comparisonTitle):
	'''
	caminhoBase:  		caminho absoluto ate a base do dataset
	gtFiel: 	  		nome do arquivo que guarda o groundTruth
	outputFile:   		nome do arquivo de saida do tracker
	rootOutput:   		nome da pasta de logs, provavelmente: "__log__"
	comparisonTitle:	nome da comparacao, geralmente Tracker + parametro, e.g.: TLD_IoU
	'''
	for root, dirs, files in os.walk(caminhoBase):
		if gtFile in files:
			determineAndWriteIoU(root,gtFile,outputFile,rootOutput,comparisonTitle)
			
def readFileAsAList(fileAbsPath):
	arquivo = []
	with open(fileAbsPath,'r') as file:
		arquivo = file.readlines()
	return arquivo
	

def removeCarriageReturn(lista):

	noCarriege = []
	
	if type(lista[0]) == list: # se for bidimensional
		for linha in lista:
			noCarriege.append(linha.replace('\n',''))
	
	else: # se for unidimensional
		if type(lista) == list:
			for linha in lista:
				noCarriege.append(linha.replace('\n',''))

	return noCarriege


def discreteIoU(iou,occlusionTruth,threshold):
	'''
	Entrada: 
			* Valores do iou, valores continuos
			* Lista de oclusa de acordo com o dataset
			* Threshold para determinacao do IoU
			OBS: Passar lista como Number e nao como string
	Saida:
			* Lista de saida dos Iou discretizada
			0:   impacto negativo
			1:   impacto positivo
			nan: neutro - ignorado
			* Valor do Recall propriamente dito
	'''
	discrete_iou = []
	for intersection, occlusion in zip(iou,occlusionTruth):

		if math.isnan(intersection) and occlusion == 0 :
			discrete_iou.append(0)

		if math.isnan(intersection)  and occlusion == 1 :
			discrete_iou.append(float('nan'))

		if (not math.isnan(intersection)) and occlusion == 0 :
			if intersection >= threshold:
				discrete_iou.append(1)
			else:
				discrete_iou.append(0)
		if (not math.isnan(intersection)) and occlusion == 1 :
			discrete_iou.append(0)
			
	return discrete_iou


def precision(discrete_iou,iou,occlusionList):
	'''
	Entrada: 
			* Uma lista de intersection over Union ja julgado entre o gt e a saida do tracker;
			* Valor do iou, valores continuos
			* Lista de oclusa de acordo com o dataset
			OBS: Passar lista como Number e nao como string
	Saida:
			* Lista de saida dos elementos que compuseram o Recall
			0:   impacto negativo
			1:   impacto positivo
			nan: neutro - ignorado
			* Valor do Recall propriamente dito
	'''
	aux_precision = []
	#print('occlusionList: ',occlusionList)
	#print('iou: ', iou)
	#print('discrete_iou: ', discrete_iou)
	for occlusion, intersection, discrete in zip(occlusionList, iou, discrete_iou):
		if math.isnan(intersection): # nao entra na conta
			aux_precision.append(float('nan'))
		if (not math.isnan(intersection)) and occlusion == 1: # penalizacao
			aux_precision.append(0)
		if (not math.isnan(intersection)) and occlusion == 0: # normal
			aux_precision.append(discrete)
	#print('aux_precision: ', aux_precision)
	finalList = [elemento for elemento in aux_precision if not math.isnan(elemento)]
	#print('finalList: ', finalList)
	val_precision = float(sum(finalList))/float(len(finalList))
	return aux_precision, float(val_precision)


def recall(discrete_iou,occlusionList):
	'''
	Entrada: 
			* Uma lista de intersectio over Union ja julgado entre o gt e a saida do tracker;
			* Lista de oclusa de acordo com o dataset
			OBS: Passar lista como Number e nao como string
	Saida:
			* Lista de saida dos elementos que compuseram o Recall
			0:   impacto negativo
			1:   impacto positivo
			nan: neutro - ignorado
			* Valor do Recall propriamente dito
	'''
	aux_recall = []

	for discrete, occlusion in zip(discrete_iou,occlusionList):
		
		if occlusion == 1 :#if occlusion == float('nan') :
			aux_recall.append(float('nan'))
		else:
			if discrete == 1:
				aux_recall.append(1)
			else:
				aux_recall.append(0)

	finalList = [elemento for elemento in aux_recall if not math.isnan(elemento)]
	val_recall = float(sum(finalList))/float(len(finalList))

	return finalList,float(val_recall)

def fMeasure(precision,recall):
	val_fMeasure = 0
	try:
		val_fMeasure = float(2*precision*recall)/float(precision+recall)
	except ZeroDivisionError:
		pass
	return val_fMeasure



PATH_DATASET 	= lp.getInJson('tracker','datasetPath')
GT_FILE 		= lp.getInJson('tracker','groundTruthFile')
TRACKER_OUTPUT 	= lp.getInJson('process','nome_saida')
OCCLUSION_FILE 	= lp.getInJson('analise','occlusion.label')
LOG_FOLDER 		= lp.getInJson('tracker','log_folder')
THRESHOLD 		= lp.getInJson('analise','threshold')

list_video 	   = []
list_precision = []
list_recall    = []
list_fMeasure  = []

#performance.append(['video','precision','recall','fMeasure'])


entrada = input('Qual o padra de entrada do groundtruth? \n\n 0) ltrb - 2 pontos \n 1) ltrb - 4 pontos \n 2) yxwh\n')
assert(int(entrada) in [0,1,2])



for root, dirs, files in os.walk(PATH_DATASET):
	
	

	
	if LOG_FOLDER in dirs:
		
		if lp.getInJson('sistema','SO') == 'windows':
			video = root.split('\\')
		else:
			video = root.split('/')
		try:
			gtList = readFileAsAList(os.path.join(root,GT_FILE))			
			gtList = removeCarriageReturn(gtList)
			outputList = readFileAsAList(os.path.join(root,LOG_FOLDER ,TRACKER_OUTPUT))
			outputList = removeCarriageReturn(outputList)
			#nao utilizado

			occlusionList = readFileAsAList(os.path.join(root,OCCLUSION_FILE))	
			occlusionList = removeCarriageReturn(occlusionList)
			occlusionList = [int(i) for i in occlusionList]


			iouList = determineIoU(GT_FILE, os.path.join(LOG_FOLDER, TRACKER_OUTPUT),root,entrada)
			

			discrete_iou = discreteIoU(iouList,occlusionList,THRESHOLD)
			precisionList,precisionValue = precision(discrete_iou,iouList,occlusionList)
			recallList,recallValue = recall(discrete_iou,occlusionList)			

			val_fMeasure = fMeasure(precisionValue,recallValue)
			
			list_video.append(video[-1])
			list_precision.append(precisionValue)
			list_recall.append(recallValue)
			list_fMeasure.append(val_fMeasure)

		except FileNotFoundError:
			
			print('erro ao processar o video: ' +video[-1])
			list_video.append(video[-1])
			list_precision.append('-----')
			list_recall.append('-----')
			list_fMeasure.append('-----')
			#performance.append['nao processado']
#print(performance)	
tabela = pd.DataFrame({"video": list_video,"precision": list_precision, "recall": list_recall, "fMeasure": list_fMeasure, })
print(tabela)
print(" ")
tabela2 = tabela.sort_values("video",na_position="first")
print(tabela2)




writer = pd.ExcelWriter(TRACKER_OUTPUT+' threshold '+ str(THRESHOLD) + ' .xlsx')#, engine = 'xlsxwriter')
tabela2.to_excel(writer,TRACKER_OUTPUT, index = False)
workbook = writer.book
#worksheet = writer.sheets['Planilha 1']
writer.save()



''' 		DADOS PARA TESTE
gtList = readFileAsAList(os.path.join(PATH_DATASET,'bag',GT_FILE))	
gtList = removeCarriageReturn(gtList)
outputList = readFileAsAList(os.path.join(PATH_DATASET,'bag',TRACKER_OUTPUT))
outputList = removeCarriageReturn(outputList)
iouList = determineIoU(GT_FILE,TRACKER_OUTPUT,os.path.join(PATH_DATASET,'bag'))
occlusionList = readFileAsAList(os.path.join(PATH_DATASET,'bag',OCCLUSION_FILE))
occlusionList= removeCarriageReturn(occlusionList)
occlusionList = [int(i) for i in occlusionList]
discrete_iou = discreteIoU(iouList,occlusionList,THRESHOLD)
_, precisionValue = precision(discrete_iou,iouList,occlusionList)
_, recallValue = recall(discrete_iou,occlusionList)
val_fMeasure = fMeasure(precisionValue,recallValue)
'''
