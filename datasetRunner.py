import os
import multiprocessing
import math
import localParameters as lp


NUM_THREAD 			= int(lp.getInJson('datasetRunner','threads'))
LISTA_DE_PARAMETROS = lp.getInJson('datasetRunner','parametros')
CAMINHO_VOT_2015 	= str(lp.getInJson('tracker','datasetPath'))
PATH_SCRIPT 		= str(lp.getInJson('tracker','trackerPath'))
NOME_ARQUIVO_SAIDA 	= str(lp.getInJson('process', 'nome_saida'))
BASH_PYTHON 		= str(lp.getInJson('sistema','python'))
LOG_FOLDER 			= lp.getInJson('tracker','log_folder')
LISTA_DE_MODOS		= lp.getInJson('datasetRunner', 'mode')

def get_list_videos(parametro):
	listVideos = []
	for i in os.listdir(CAMINHO_VOT_2015):
		if(not i.startswith('_')) and (os.path.isdir(os.path.join(CAMINHO_VOT_2015,i))):
			if(not ((NOME_ARQUIVO_SAIDA+str(parametro)) in os.listdir(os.path.join(CAMINHO_VOT_2015,i,LOG_FOLDER)))):
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

def runner(id, list_video, parametro, modo):
	list_video_finished = [False] * len(list_video)

	for videoName in list_video:
		nomeCompleto = NOME_ARQUIVO_SAIDA + str(parametro) + '_modo_' + str(modo) # identifica o parametro que foi executado cada um dos arquivo como um sufixo
		try:
			os.mkdir(os.path.join(CAMINHO_VOT_2015,videoName,LOG_FOLDER))
		except:
			pass
		finally:
			os.system(' '.join([BASH_PYTHON, PATH_SCRIPT,"-v", videoName,"-n", nomeCompleto, "-m", str(modo), "-p",str(parametro)]))


def main():
	for parametro in LISTA_DE_PARAMETROS:
		for modo in LISTA_DE_MODOS:
			listVideos = get_list_videos(parametro)
			part_list_videos = get_new_list_video(listVideos, NUM_THREAD)

			if(len(part_list_videos[0]) != 0):
				list_process = []
				for id_thread in range(NUM_THREAD):
					p = multiprocessing.Process(target=runner, args=(id_thread, part_list_videos[id_thread], parametro, modo))
					list_process.append(p)
					p.start()

				for p in list_process:
					p.join()


if __name__ == '__main__':
	main()
