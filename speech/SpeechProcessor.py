import os
import torch
import whisper
from groq import Groq
import sounddevice as sd
import wave
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from pynput import keyboard
import shutil
import threading

from dotenv import load_dotenv
load_dotenv()

class SpeechProcessor:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #TTS
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.tts_type = "playai-tts"
        self.voice = "Basil-PlayAI"
        self.response_format = "wav"
        
        #STT
        self.model = whisper.load_model(model_size).to(self.device)
        
        self.samplerate = 44100
        self.recording = []
        self.is_recording = False
        self.listener = None
        self.record_thread = None
        self.__check_ffmpeg()

    def __check_ffmpeg(self):
        """Check if ffmpeg is installed for pydub."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found! Install it using: sudo apt install ffmpeg")

    def __convert_to_wav(self, input_file, output_file=None):
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File '{input_file}' not found!")
        
        if output_file is None:
            output_file = os.path.splitext(input_file)[0] + ".wav"
        
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        
        print(f"Converted '{input_file}' to '{output_file}'")
        return output_file

    #PUBLIC
    def text_to_speech(self, text, is_groq=False, output_file="output.wav"):
        """Convert given text to audio file:
        - 

        Args:
            text (_type_): _description_
            output_file (str, optional): _description_. Defaults to "output.wav".
        """
        if is_groq:
            response = self.client.audio.speech.create(
            model=self.tts_type,
            voice=self.voice,
            input=text,
            response_format=self.response_format
            )

            response.write_to_file(output_file)
            
        else:
            tts = gTTS(text)
            temp_mp3 = "temp.mp3"
            tts.save(temp_mp3)
            output_file = self.__convert_to_wav(temp_mp3, output_file)
            os.remove(temp_mp3)
        
        print(f"TTS saved as {output_file}")
        return output_file
    
    #PUBLIC
    def play_audio(self, file_path):
        """Play audio file from a given path, then remove that audio file

        Args:
            file_path (_str_): *path to the audio file*
        """
        if not os.path.exists(file_path):
            print(f"File '{file_path}' not found!")
            return
        audio = AudioSegment.from_wav(file_path)
        play(audio)
        #os.remove(file_path)
        print(f"Deleted: {file_path}")

    def __audio_callback(self, indata, frames, time, status):
        """This is called for each audio block."""

        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            self.recording.append(indata.copy())

    def __start_recording_thread(self):
        """Start the actual audio recording in a separate thread."""
        self.recording = []  # Clear any previous recordings
        
        def record_thread_func():
            with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.__audio_callback):
                while self.is_recording:
                    sd.sleep(100)  # Sleep to reduce CPU usage
        
        self.record_thread = threading.Thread(target=record_thread_func)
        self.record_thread.start()

    def __on_key_press(self, key):
        if key == keyboard.Key.space:
            self.is_recording = not self.is_recording
            
            if self.is_recording:
                print("Recording... Press SPACE to stop.")
                self.__start_recording_thread()
            else:
                print("Recording stopped.")
                self.listener.stop()
                
    def __save_recording(self, file_path):
        if not self.recording:
            print("No audio recorded.")
            return
        
        audio_data = np.concatenate(self.recording, axis=0)
        
        # Convert to int16 format for wave file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_int16.tobytes())

    #PUBLIC
    def record_audio(self, output_file="recorded.wav"):
        """Record audio, suing spacebar to start and stop!

        Args:
            output_file (str, optional): Path to the output audio file recorded. Defaults to "recorded.wav".

        Returns:
            _str_: The output file path, if not specified
        """
        print("Press SPACE to start recording...")
        self.listener = keyboard.Listener(on_press=self.__on_key_press)
        self.listener.start()
        self.listener.join()
        
        # Make sure the recording thread is finished
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join()
            
        self.__save_recording(output_file)
        print(f"Recording saved as {output_file}")
        return output_file
            
    #PUBLIC
    def speech_to_text(self, audio_file):
        """Use a STT model to return corresponding text for a give audio file

        Args:
            audio_file (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = self.model.transcribe(audio_file)
        return result["text"]


if __name__ == "__main__":
    sp = SpeechProcessor()
    sp.text_to_speech("This is a test prompt. Harry Potter is a wizard. Do you agree Mukesh?", output_file="output.wav")
    sp.play_audio("output.wav")
    #recorded_file = sp.record_audio("my_recording.wav")
    #transcription = sp.speech_to_text(recorded_file)
    #print(f"Transcription: {transcription}")