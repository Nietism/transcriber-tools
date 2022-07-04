import auditok
import os
import csv

import paddle
from paddlespeech.cli import ASRExecutor, TextExecutor
import warnings
warnings.filterwarnings('ignore')

asr_executor = ASRExecutor()
text_executor = TextExecutor()

def audio2txt(path):

    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[:3]))
    words = []
    for file in filelist:
        print(path+'/'+file)
        text = asr_executor(
            audio_file=path+'/'+file,
            device=paddle.get_device())
        if text:
            result = text_executor(
                text=text,
                task='punc',
                model='ernie_linear_p3_wudao',
                device=paddle.get_device())
        else:
            result = text
        words.append(result)
    return words


def txt2csv(txt):
    with open('SXY_asr_result.csv', 'w', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        for row in txt:
            f_csv.writerow([row])


def audio_segment(
        path, 
        ty='video', 
        mmin_dur=1, 
        mmax_dur=100000, 
        mmax_silence=1, 
        menergy_threshold=55
    ):

    file = path
    audio_regions = auditok.split(
        file,
        min_dur=mmin_dur,  # minimum duration of a valid audio event in seconds
        max_dur=mmax_dur,  # maximum duration of an event
        # maximum duration of tolerated continuous silence within an event
        max_silence=mmax_silence,
        energy_threshold=menergy_threshold  # threshold of detection
    )

    for i, r in enumerate(audio_regions):
        # Regions returned by `split` have 'start' and 'end' metadata fields
        print(
            "Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))

        epath = ''
        file_pre = str(epath.join(file.split('.')[0].split('/')[-1]))

        mk = './audio_segments'
        if (os.path.exists(mk) == False):
            os.mkdir(mk)
        if(os.path.exists(mk + '/' + ty) == False):
            os.mkdir(mk + '/' + ty)
        if(os.path.exists(mk + '/' + ty + '/' + file_pre) == False):
            os.mkdir(mk + '/' + ty + '/' + file_pre)

        num = i
        
        s = '000000' + str(num)

        file_save = mk + '/' + ty + '/' + file_pre + '/' + \
            s[-3:] + '-' + '{meta.start:.3f}-{meta.end:.3f}' + '.wav'

        filename = r.save(file_save)
        print("region saved as: {}".format(filename))

    return mk + '/' + ty + '/' + file_pre




if __name__ == "__main__":
    path = audio_segment(path='./SXY.wav', ty='audio', mmin_dur=0.5, mmax_dur=100000, mmax_silence=0.5, menergy_threshold=55)
    txt_all = audio2txt(path)
    txt2csv(txt_all)








    