from __future__ import unicode_literals
"""

このコードはSpeech2gestureのREADME.mdがあるフォルダで行なうこと

"""
import argparse
from subprocess import call

import cv2
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm
import youtube_dl
from yt_dlp import YoutubeDL
parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

BASE_PATH = args.base_path
df = pd.read_csv(os.path.join(BASE_PATH, "videos_links.csv"))

if args.speaker:
    df = df[df['speaker'] == args.speaker]

temp_output_path = 'temp.mp4'

for _, row in tqdm(df.iterrows(), total=df.shape[0]):

    i, name, link = row
    if 'youtube' in link:
        try:
            output_path = os.path.join(BASE_PATH, row["speaker"], "videos", row["video_fn"])
            if not (os.path.exists(os.path.dirname(output_path))):
                os.makedirs(os.path.dirname(output_path))
            #command = 'youtube-dl -o {temp_path} -f mp4 {link}'.format(link=link, temp_path=temp_output_path)
            #res1 = call(command, shell=True)
            ydl_opts = {'output': temp_output_path, "format":"mp4", "outtmpl": temp_output_path}
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.download([link])
            #print(result)
            #call("touch temp.mp4",shell = True)
            call("chmod 777 temp.mp4", shell = True)
            #res1 = call('find . -type f -name "*.mp4" -exec sh -c "mv "$0" "temp.mp4"" {} \;', shell=True)
            #print(res1)

            cam = cv2.VideoCapture(temp_output_path)
            if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
                shutil.move(temp_output_path, output_path)
            else:
                res2 = call('temp.mp4 ffmpeg -i "%s" -r 30000/1001 -strict -2 "%s" -y' % (temp_output_path, output_path),
                            shell=True)
        except Exception as e:
            print(e)
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
print("Out of a total of %s videos for %s: "%(len(df), args.speaker))
print("Successfully downloaded:")
my_cmd = 'ls ' + os.path.join(BASE_PATH, row["speaker"], "videos") + ' | wc -l'
os.system(my_cmd)
