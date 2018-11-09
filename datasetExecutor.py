import os

CAMINHO_VOT2015 = '/home/swhants/Documentos/vot2015/'
listVideos = [ i for i in os.listdir(CAMINHO_VOT2015) if (not i.startswith('_')) and (os.path.isdir(os.path.join(CAMINHO_VOT2015,i)))]
listVideos.sort()
print(listVideos)

fileName = 'erro' #_acumulada_6_por_6_individual'

for cont, videoName in enumerate(listVideos):
	if cont < 41:
		continue

	if 'siameseFC_media_acumulada30' in os.listdir(os.path.join(CAMINHO_VOT2015,videoName,'__log__')):
		print('skiping: ', videoName, '...')
		pass
	else:		
		try:
			os.mkdir(os.path.join(CAMINHO_VOT2015,videoName,'__log__'))
			print('Video ',cont+1,' de: ',len(listVideos))
			print('videoName: ', videoName)
			os.system('python3.6 tracker.py --video '+ videoName)
		except:
			print('Video ',cont+1,' de: ',len(listVideos))
			print('videoName: ', videoName)
			os.system('python3.6 tracker.py --video '+ videoName)