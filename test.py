import json
import socket
import localParameters as lp

NUM_THREAD = lp.getInJson('datasetRunner','threads')
LISTA_DE_PARAMETROS = lp.getInJson('datasetRunner','parametros')
CAMINHO_VOT_2015 = lp.getInJson('sistema','datasetPath')
PATH_SCRIPT = lp.getInJson('tracker','trackerPath')
NOME_ARQUIVO_SAIDA = lp.getInJson('datasetRunner', 'nome_saida')


'''
j = lp.getInJson("sistema","datasetPath")

def readParameters():
	with open('parameter.json') as file:
		k = json.load(file)
	return k


k = readParameters()

print('sistema' in k[1].keys())
'''

'''
def readParameters():
	with open("parameter.json") as file:
		k = json.load(file)


	for elemento in k:
		
		if elemento['sistema']['computador'] ==  socket.gethostname():
			caminhoDataset = elemento['sistema']['datasetPath']
			print(caminhoDataset)
			trackerPath = elemento['tracker']['trackerPath']
			inAServer = elemento['tracker']['show']
			print(inAServer)

			return caminhoDataset, trackerPath,inAServer
	assert False, 'Nao ha parametros definidos para rodar esse escript nessa maquina.'
'''

'''
def readParameters():
	with open("datasetRunner.json") as file:
		k = json.load(file)


	for elemento in k:
		
		print(elemento['datasetRunner']['parametros'])
'''


