import wave
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

# Configurações do Áudio e Ficheiro de Saída
FORMAT = pyaudio.paInt16  # 16-bit
CHANNELS = 1  # Mono
CHUNK = 2048  # Tamanho do bloco
OUTPUT_FILENAME = "audio_gravado.wav"  # Nome do ficheiro WAV gerado

# Inicializar o PyAudio
p = pyaudio.PyAudio()

# --- Deteção Automática do Dispositivo de Microfone ---
try:
    default_input_info = p.get_default_input_device_info()
    DEVICE_INDEX = default_input_info["index"]
    RATE = int(default_input_info["defaultSampleRate"])
    print(
        f"Microfone encontrado: {default_input_info['name']} (ID: {DEVICE_INDEX}, Rate: {RATE}Hz)"
    )
except IOError:
    print(
        "ERRO: Nenhum microfone padrão ativo encontrado. Verifica o Painel de Controlo do Windows."
    )
    p.terminate()
    exit()

# Abrir o stream de áudio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)

# Lista para armazenar todos os blocos de áudio gravados
frames = []

# Criar a janela de Hanning
window = np.hanning(CHUNK)

# Criar a figura com 2 subplots
fig, (ax_wave, ax_fft) = plt.subplots(2, 1, figsize=(10, 6))
fig.tight_layout(pad=3.0)

# --- Subplot 1: Forma de Onda (Tempo) ---
x_wave = np.arange(0, CHUNK)
(line_wave,) = ax_wave.plot(x_wave, np.zeros(CHUNK), color="b")
ax_wave.set_title("1. Forma de Onda (Tempo)")
ax_wave.set_xlabel("Amostras")
ax_wave.set_ylabel("Amplitude (int16)")
ax_wave.set_ylim(-32768, 32767)
ax_wave.set_xlim(0, CHUNK)
ax_wave.grid(True)

# --- Subplot 2: Espectro de Frequências (FFT) ---
xf = np.fft.rfftfreq(CHUNK, 1 / RATE)
(line_fft,) = ax_fft.plot(xf, np.zeros(len(xf)), color="r")
ax_fft.set_title("2. Espectro de Frequências (FFT)")
ax_fft.set_xlabel("Frequência (Hz)")
ax_fft.set_ylabel("Magnitude Normalizada")
ax_fft.set_xlim(0, 5000)  # Foco na faixa de voz (0-5000 Hz)
ax_fft.set_ylim(0, 1.0)
ax_fft.grid(True)

print("A gravar... Fala para o microfone. Fecha a janela para guardar e sair.")

try:
    while plt.fignum_exists(fig.number):
        # Ler dados do microfone
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)  # Armazenar os bytes do áudio na memória

        data_int = np.frombuffer(data, dtype=np.int16)

        # Normalizar o sinal para a faixa [-1.0, 1.0]
        data_normalized = data_int / 32768.0

        # Atualizar gráfico da onda
        line_wave.set_ydata(data_int)

        # Aplicar janela e calcular a FFT
        fft_data = np.abs(np.fft.rfft(data_normalized * window)) / (CHUNK / 2)

        # Atualizar gráfico de frequência
        line_fft.set_ydata(fft_data)

        # Ajuste dinâmico suave da escala Y
        max_val = np.max(fft_data)
        current_ylim = ax_fft.get_ylim()[1]
        if max_val > current_ylim:
            ax_fft.set_ylim(0, max_val * 1.2)

        # Redesenhar a interface
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("Gravação interrompida pelo utilizador.")

finally:
    # Encerrar a captura de áudio
    stream.stop_stream()
    stream.close()
    sample_size = p.get_sample_size(FORMAT)
    p.terminate()

    # --- Guardar o áudio capturado num ficheiro .WAV ---
    if frames:
        print(f"\nA guardar ficheiro WAV em '{OUTPUT_FILENAME}'...")
        wf = wave.open(OUTPUT_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_size)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        print(f"Sucesso! Ficheiro '{OUTPUT_FILENAME}' guardado na mesma pasta do script.")
    else:
        print("Nenhum dado de áudio foi gravado.")