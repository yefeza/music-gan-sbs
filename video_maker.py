import moviepy.editor as mp
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Ruta del archivo de audio
audio_path = "for_videos/1.wav"

# Cargar el audio utilizando librosa
audio, sr = librosa.load(audio_path)

# Configuración de la animación
duration = librosa.get_duration(y=audio, sr=sr)
fps = 30

# Generar los valores de las ondas de sonido
times = np.linspace(0, duration, int(duration * fps))
amplitudes = np.interp(times, np.linspace(0, duration, len(audio)), audio)

# Crear una función para generar el video con la animación de ondas
def generate_video_with_waves(t):
    #convertir el tiempo en segundos a frames
    t = int(t * fps)
    from_time = max(0, t - fps)
    to_time = t
    # Crear la figura
    fig, ax = plt.subplots()
    # set background color to black
    fig.patch.set_facecolor('black')
    # set axes color to black
    ax.set_facecolor('black')
    # set axes limits
    ax.plot(times[from_time:to_time], amplitudes[from_time:to_time], color='white', marker='o', linestyle='solid', markersize=1)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

# Crear el clip de video
clip = mp.VideoClip(generate_video_with_waves, duration=duration)

# Establecer el audio en el clip de video
audio_clip = mp.AudioFileClip(audio_path)
clip = clip.set_audio(audio_clip)

# Escribir el video a un archivo
output_video_path = "for_videos/1.mp4"
clip.write_videofile(output_video_path, codec="libx264", fps=fps)