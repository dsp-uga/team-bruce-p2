import os
import subprocess
import tarfile
import shutil


class DataLoader:
	"""
	Downloads data from Google Storage bucket, if not already in the folder.
	After downloading, it untars the data, puts the train video files into 
	'train' folder and test video files in the 'test' folder. Furthermore, 
	the frames in the train, test videos are returned as numpy arrays, in a
	Pandas dataframe alongwith their hashes.

	Arguments
	---------
	url : string
		Google Cloud Storage bucket URL containing the Cilia data

	"""
	def __init__(self, url="gs://uga-dsp/project2"):
		"""
		Creates a loader for Cilia data

		Arguments
		---------
		url: string
			Google Cloud Storage bucket URL containing the Cilia data
		"""
		bucket_url = url
		self.dataset_folder = '../cilia_dataset'
		self.cwd = os.getcwd()
		if(os.path.isdir(self.dataset_folder)):
			pass
			self.setup_data()
		else:
			self.download(bucket_url)
			self.setup_data()


	def download(self, bucket_url):
		"""
		Downloads Cilia data from Google Storage bucket

		Arguments
		---------
		bucket_url: string
			Google Cloud Storage bucket URL containing the Cilia data
		"""
		os.mkdir(self.dataset_folder)
		os.chdir(self.dataset_folder)
		subprocess.call('google-cloud-sdk/bin/gsutil -m cp -r ' + bucket_url + '/*',  shell=True)
		os.chdir(self.cwd)

		
	def setup_data(self):
		"""
		Processes the downloaded data in the 'cilia_data' folder so as
		to untar the tar files in the 'data' folder, move the train, test
		video files to 'train' and 'test' respectively, and to clean the 
		remaining tar files.
		"""
		self.train_hashes = self.read_file(self.dataset_folder + '/train.txt')
		self.test_hashes = self.read_file(self.dataset_folder + '/test.txt')
		# Extract train tar files into 'train' folder
		os.mkdir(self.dataset_folder + '/train') if not(os.path.isdir(self.dataset_folder + '/train')) else None
		for train_hash in self.train_hashes:
			tar = tarfile.open(os.path.join(self.dataset_folder + '/data', str(train_hash) + '.tar'))
			tar.extractall(self.dataset_folder + "/train/")
			tar.close()
		# Extract test tar files into 'test' folder
		os.mkdir(self.dataset_folder + '/test') if not(os.path.isdir(self.dataset_folder + '/test')) else None
		for test_hash in self.test_hashes:
			tar = tarfile.open(os.path.join(self.dataset_folder + '/data', str(test_hash) + '.tar'))
			tar.extractall(self.dataset_folder + "/test/")
			tar.close()
		# Remove tar files from 'cilia_dataset' directory
		files_in_directory = os.listdir(self.dataset_folder)
		for item in files_in_directory:
			if item.endswith(".tar"):
				os.remove(os.path.join(self.dataset_folder, item))
		shutil.rmtree(self.dataset_folder + '/data')


	def read_file(self, file_name):
		"""
		Reads the text files containing hashes and returns the hashes
		in a list

		Arguments
		---------
		file_name : string
			Name of the files containing train hashes and test hashes

		Returns:
		--------
		list:
			List containing hashes	
		"""
	    f = open(file_name, "r")
	    return f.read().split()    
		