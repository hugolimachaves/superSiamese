'''
transforma um arquivo com coordenadas poligonais - 8 pontos - em um arquivo yxwh
'''

import numpy as np

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

	return [cx, cy, w, h]

def abrir(filePath):
	with open(filePath,'r') as file:
		lines = file.readlines()
		return lines

def changeFormat(lines):
	newLines = []
	for linha in lines:
		region = [float(i) for i in linha.strip().split(",")]
		cx, cy, w, h = getAxisAlignedBB(region)
		newLine = str(int(cx)) + ',' + str(int(cy)) + ',' + str(int(w)) + ',' + str(int(h)) + '\n' 
		newLines.append(newLine)
	return newLines

def escrever(filePath,newLines):
	with open(filePath+'.TLD','w') as outfile:
		outfile.writelines(changeFormat(newLines))




#escrever('groundtruth.txt',abrir('groundtruth.txt'))
	
'''
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
'''