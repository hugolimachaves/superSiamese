import os
import psutil
import time
import random
from multiprocessing import Pool
import multiprocessing
import videoMemory as vm

MARGEM_GIGAS_LIVRES = 4
NUM_THREAD = 24

listaDeParametros = [0.1,0.001,0.05,0.005,1]

CAMINHO_VOT2015 = '/home/schrodinger/sdc/hugo/vot2015/'
#CAMINHO_VOT2015 = '/home/swhants/Documentos/vot2015/'
#assert 1==2,'modificar o que esta abaixo desse assert'
PATH_SCRIPT = '/home/schrodinger/sdc/hugo/superSiamese/tracker.py'
#PATH_SCRIPT = '/home/swhants/Documentos/superSiamese-master/tracker.py'
NOME_ARQUIVO_SAIDA = 'filtro_adaptativo_mi_'
listVideos = [ i for i in os.listdir(CAMINHO_VOT2015) if (not i.startswith('_')) and (os.path.isdir(os.path.join(CAMINHO_VOT2015,i)))]
listVideos.sort()


### Divisao da lista de videos em N_PARTES que eh o num de threads
N_PARTES = NUM_THREAD

def get_new_list_video():
	four_list_videos = []
	aux = []
	cont = 0

	for video in listVideos:
		if cont == int(len(listVideos) / N_PARTES):
			four_list_videos.append(aux)
			aux = []
			cont = 0

		aux.append(video)
		cont += 1

	four_list_videos.append(aux)
	return four_list_videos

four_list_videos = get_new_list_video()
### ~Divisao da lista de videos em N_PARTES



def has_finished(list_video_finished):
	for i in list_video_finished:
		if(not i):
			return False
	return True

def runner( id, cont, list_video, parametro):
	list_video_finished = [False] * len(list_video)

	i = 0
	while(True):
		if(i < len(list_video)):
			if(not list_video_finished[i]):
				videoName = list_video[i]
				if (NOME_ARQUIVO_SAIDA+str(parametro)) in os.listdir(os.path.join(CAMINHO_VOT2015,videoName,'__log__')):
					list_video_finished[i] = True
					i += 1

				else:
					try:
						os.mkdir(os.path.join(CAMINHO_VOT2015,videoName,'__log__'))
						print('videoName: ', videoName)
						os.system('python3.6 ' + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' ' + CAMINHO_VOT2015 + ' ' + str(parametro))

					except:
						print('videoName: ', videoName)
						os.system('python3.6  ' + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' '  + CAMINHO_VOT2015 + ' ' +str(parametro) )

					list_video_finished[i] = True
			
			else:
				if(has_finished(list_video_finished)):
					break
				i += 1
		else:
			i = 0

def main():
	for parametro in listaDeParametros:

		cont = multiprocessing.Value('i',0)
		finish = multiprocessing.Value('i',0)
		free_memory = multiprocessing.Value('d',0.0)
		pos_list_video = multiprocessing.Array('i',range(len(listVideos)))

		list_process = []
		for id_thread in range(NUM_THREAD):
			p = multiprocessing.Process(target=runner, args=( id_thread, cont, four_list_videos[id_thread],  parametro))
			list_process.append(p)
			p.start()

		for p in list_process:
			p.join()


if __name__ == '__main__':
	main()