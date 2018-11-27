import json
import socket


def readParameters():
	with open("parameter.json") as file:
		k = json.load(file)
	return k


	for elemento in k:
		
		if elemento['sistema']['computador'] ==  socket.gethostname():
			caminhoDataset = elemento['sistema']['datasetPath']
			trackerPath = elemento['tracker']['trackerPath']
			inAServer = elemento['tracker']['show']
			print(inAServer)

			return caminhoDataset, trackerPath,inAServer
	assert False, 'Nao ha parametros definidos para rodar esse escript nessa maquina.'

def getInJson(parametro1, parametro2):
	pyson = readParameters():
	for elemento in pyson:
		if elemento['sistema']['computador'] ==  socket.gethostname():
			return elemento[parametro1][parametro2]
		else
			return False