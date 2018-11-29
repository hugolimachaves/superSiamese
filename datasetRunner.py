import os
import multiprocessing
import math
import localParameters as lp


NUM_THREAD = int(lp.getInJson('datasetRunner','threads'))
LISTA_DE_PARAMETROS = lp.getInJson('datasetRunner','parametros')
CAMINHO_VOT_2015 = str(lp.getInJson('sistema','datasetPath'))
PATH_SCRIPT = str(lp.getInJson('tracker','trackerPath'))
NOME_ARQUIVO_SAIDA = str(lp.getInJson('datasetRunner', 'nome_saida'))
BASH_PYTHON = str(lp.getInJson('sistema','python'))

def get_list_videos(parametro):
	listVideos = []
	for i in os.listdir(CAMINHO_VOT_2015):
		if(not i.startswith('_')) and (os.path.isdir(os.path.join(CAMINHO_VOT_2015,i))):
			if(not ((NOME_ARQUIVO_SAIDA+str(parametro)) in os.listdir(os.path.join(CAMINHO_VOT_2015,i,'__log__')))):
				listVideos.append(i)
	listVideos.sort()
	return listVideos

def get_new_list_video(listVideos, n_partes):
	new_list_videos = []
	aux = []
	cont = 0

	for video in listVideos:
		if cont == math.ceil(len(listVideos) / n_partes):
			new_list_videos.append(aux)
			aux = []
			cont = 0

		aux.append(video)
		cont += 1

	new_list_videos.append(aux)

	return new_list_videos

def runner(id, list_video, parametro):
	list_video_finished = [False] * len(list_video)

	for videoName in list_video:
		try:
			os.mkdir(os.path.join(CAMINHO_VOT_2015,videoName,'__log__'))
			os.system(BASH_PYTHON + ' '  + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' ' + CAMINHO_VOT_2015 + ' ' + str(parametro))
		except:
			os.system(BASH_PYTHON +' ' + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' '  + CAMINHO_VOT_2015 + ' ' +str(parametro))

def main():

	for parametro in LISTA_DE_PARAMETROS:

		listVideos = get_list_videos(parametro)
		part_list_videos = get_new_list_video(listVideos, NUM_THREAD)

		for i in part_list_videos:
			print(i)

		list_process = []
		for id_thread in range(NUM_THREAD):
			p = multiprocessing.Process(target=runner, args=(id_thread, part_list_videos[id_thread],  parametro))
			list_process.append(p)
			p.start()


		for p in list_process:
			p.join()

def _get_Args():
	parser = argparse.ArgumentParser()
	parser.add_argument('video', help= 'nome da pasta do video')
	parser.add_argument('nome', help = 'nome do arquivo de saida')
	parser.add_argument('caminho', help ='caminho ABSOLUTO para o dataset')
	parser.add_argument('parametro', help = 'parametro a ser setado para esse tracker')
	return parser.parse_args()


if __name__ == '__main__':
	main()
