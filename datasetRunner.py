import os
import psutil
import time
import random
from multiprocessing import Pool
import multiprocessing
import videoMemory as vm


listaDeParametros = [0.01,0.1,0.001,0.05,0.005,1]

CAMINHO_VOT2015 = '/home/hugo/Documents/Mestrado/vot2015/'
assert 1==2,'modificar o que esta abaixo desse assert'
PATH_SCRIPT = '/home/hugo/Documents/Mestrado/neoSiamese/tracker.py'
NOME_ARQUIVO_SAIDA = 'filtro_adaptativo_mi_'
listVideos = [ i for i in os.listdir(CAMINHO_VOT2015) if (not i.startswith('_')) and (os.path.isdir(os.path.join(CAMINHO_VOT2015,i)))]
listVideos.sort()



MARGEM_GIGAS_LIVRES = 2
NUM_THREAD = 4

free_memory = 0
finish = False

def refresh_free_memory(free_memory):
	free_memory.value = psutil.virtual_memory().available / (1024**3)

def get_free_memory(free_memory):
	return (free_memory.value - MARGEM_GIGAS_LIVRES)

def has_finished(cont_video):
	for i in range(len(listVideos)):
		if(cont_video[i] != -1):
			return 0
	return 1

def runner(lock, id, cont, free_memory, pos_list_video, finish, parametro):
	while(True):
		lock.acquire()

		if(finish.value):
			break

		time.sleep(90)		# Talvez aumentar na Xodinha

		try:
			if(cont.value < len(listVideos)):
				if(pos_list_video[cont.value] != -1):

					videoName = listVideos[cont.value]
					size_video = vm.gbPerVideo(os.path.join(CAMINHO_VOT2015,videoName))

					refresh_free_memory(free_memory)

					if size_video < get_free_memory(free_memory):

						pos_list_video[cont.value] = -1
						finish.value = has_finished(pos_list_video)

						lock.release()

						try:
							os.mkdir(os.path.join(CAMINHO_VOT2015,videoName,'__log__'))
							print('videoName: ', videoName)
							os.system('python ' + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' ' + CAMINHO_VOT2015 + str(parametro))

						except:
							print('videoName: ', videoName)
							os.system('python ' + PATH_SCRIPT + ' ' + videoName + ' ' + NOME_ARQUIVO_SAIDA + str(parametro) + ' '  + CAMINHO_VOT2015 + str(parametro) )

					else:
						lock.release()
						time.sleep(5)


				else:
					cont.value += 1
					lock.release()
					time.sleep(5)


			else:
				cont.value = 0
				lock.release()
				time.sleep(5)

		finally:
			pass

def main():

	for parametro in listaDeParametros:
		lock = multiprocessing.Lock()
		cont = multiprocessing.Value('i',0)
		finish = multiprocessing.Value('i',0)
		free_memory = multiprocessing.Value('d',0.0)
		pos_list_video = multiprocessing.Array('i',range(len(listVideos)))

		list_process = []
		for id_thread in range(NUM_THREAD):
			p = multiprocessing.Process(target=runner, args=(lock, id_thread, cont, free_memory, pos_list_video, finish, parametro))
			list_process.append(p)
			p.start()

		for p in list_process:
			p.join()


if __name__ == '__main__':
	main()