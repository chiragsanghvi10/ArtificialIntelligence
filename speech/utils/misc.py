from scipy.io import wavfile
from pathlib import Path
import shutil
import requests
import os
requests.adapters.DEFAULT_RETRIES = 50
import time
from shutil import copy2

def get_task_url(task_id):
    return 'https://storage.googleapis.com/istar-static/'+task_id+'.wav'

def split_stereo(file_path, destination_folder = 'NA'):
    fs, data = wavfile.read(file_path)            # reading the file
    file_name = Path(file_path).resolve().stem
    if destination_folder == 'NA':
        # make the destination folder as the same folder of the file
        destination_folder = os.path.dirname(file_path) + '/' + file_name + '/'
        if os.path.isdir(destination_folder):
            shutil.rmtree(destination_folder)
        os.makedirs(destination_folder)
    split_file_1 = destination_folder + file_name + '_1.wav'
    split_file_2 = destination_folder + file_name + '_2.wav'
    wavfile.write(split_file_1, fs, data[:, 0])   # saving first column which corresponds to channel 1
    wavfile.write(split_file_2, fs, data[:, 1])   # saving second column which corresponds to channel 2
    return [split_file_1, split_file_2]

def download_file(url, folder):
    success = False
    local_filename = url.split('/')[-1]
    if not os.path.exists(folder):
        os.makedirs(folder)
    print('Downloading file ' + local_filename + ' ... to '+folder)
    page = "a"
    while page == "a":
        try:
            with requests.get(url, stream=True) as r:
                page = r
                r.raise_for_status()
                with open(folder + local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            # filter out keep-alive new chunks
                            f.write(chunk)
                            # f.flush()
            break
        except:
            print(" while downloading "+url+", Connection refused by the server.. gonna sleep for 5 seconds and then retry ZZzzzz...")
            time.sleep(5)
            continue
    if os.path.exists(folder + local_filename):
        print("Successfully downloaded "+local_filename + " from url: "+url+" to folder: "+folder)
        success = True
    else:
        print("Failed to download "+local_filename + " from url: "+url+" to folder: "+folder)
    return {"abs_path": folder + local_filename, "success": success}

def reset_folders(folders):
    for folder in folders:
        if os.path.exists(folder):
            print("Deleting folder and its contents: "+folder)
            shutil.rmtree(folder)
        os.makedirs(folder)

def copy_files(file_path, destination_folder):
    if not os.path.exists(file_path):
        print('The file at path: '+file_path+' does not exist, please check')
        return
    copy2(file_path, destination_folder)

from pydub import AudioSegment

def stereo_to_mono(input, output):
    sound = AudioSegment.from_wav(input)
    sound = sound.set_channels(1)
    sound.export(output, format="wav")
