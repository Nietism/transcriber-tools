import librosa
import soundfile


if __name__ == "__main__":
    y,sr = librosa.load("./SXY.MP3")
    soundfile.write("./SXY.wav",y,sr)
    print("convert to wav format finished!")
