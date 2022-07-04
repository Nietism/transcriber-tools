import librosa
import soundfile


if __name__ == "__main__":
    y, sr = librosa.load("./SXY.MP3", sr=16000)
    # soundfile.write("./SXY.wav", y, sr)
    soundfile.write("./SXY.wav", y, 16000)
    print("convert to wav format finished!")
