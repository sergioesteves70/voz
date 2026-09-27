import os
import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class AudioComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparador de Origem e Similaridade de Áudio")
        self.root.geometry("1000x750")

        self.file_paths = []

        # --- Painel Superior: Botões de Controlo ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(
            btn_frame, 
            text="Carregar Ficheiros de Som", 
            command=self.load_files,
            font=("Arial", 10)
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_compare = tk.Button(
            btn_frame, 
            text="Comparar Similaridade (%)", 
            command=self.compare_audio_origin,
            font=("Arial", 10, "bold"),
            bg="#e1e1e1"
        )
        self.btn_compare.pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(
            btn_frame, 
            text="Selecione 2 ou mais ficheiros de áudio para começar.", 
            font=("Arial", 10)
        )
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        # --- Contentor para os Gráficos Matplotlib ---
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_files(self):
        """Abre o seletor de ficheiros para escolher múltiplos ficheiros de som."""
        paths = filedialog.askopenfilenames(
            title="Selecione ficheiros de áudio",
            filetypes=[("Ficheiros de Áudio", "*.wav *.mp3 *.flac *.ogg *.m4a")]
        )
        if paths:
            self.file_paths = list(paths)
            self.lbl_status.config(
                text=f"{len(self.file_paths)} ficheiros carregados.", 
                fg="black"
            )
            self.plot_waveforms()

    def plot_waveforms(self):
        """Desenha a forma de onda (waveform) de cada ficheiro carregado."""
        # Limpa gráficos anteriores
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        num_files = len(self.file_paths)
        if num_files == 0:
            return

        fig, axes = plt.subplots(num_files, 1, figsize=(10, max(2.5 * num_files, 4)), sharex=True)
        if num_files == 1:
            axes = [axes]

        for i, path in enumerate(self.file_paths):
            try:
                # Carrega o áudio
                y, sr = librosa.load(path, sr=None)
                
                # Renderiza a forma de onda
                librosa.display.waveshow(y, sr=sr, ax=axes[i], alpha=0.75, color="#1f77b4")
                
                file_name = os.path.basename(path)
                axes[i].set_title(f"Ficheiro {i+1}: {file_name}", fontsize=10, fontweight="bold")
                axes[i].set_ylabel("Amplitude")
                axes[i].grid(True, linestyle="--", alpha=0.5)
            except Exception as e:
                messagebox.showerror("Erro ao carregar", f"Erro ao ler {os.path.basename(path)}:\n{e}")

        axes[-1].set_xlabel("Tempo (segundos)")
        plt.tight_layout()

        # Insere o gráfico na interface do Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def calculate_similarity(path1, path2):
        """
        Calcula a percentagem de similaridade entre dois áudios
        usando a Semelhança de Cosseno sobre as características MFCC.
        """
        # Carrega os ficheiros padronizando a taxa de amostragem
        y1, sr1 = librosa.load(path1, sr=22050)
        y2, sr2 = librosa.load(path2, sr=22050)

        # Extrai os coeficientes MFCC (timbre e frequências)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

        # Calcula o vetor médio ao longo do tempo
        vec1 = np.mean(mfcc1, axis=1)
        vec2 = np.mean(mfcc2, axis=1)

        # Semelhança de Cosseno
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        cosine_sim = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

        # Mapeia a escala de [-1, 1] para [0%, 100%]
        similarity_percentage = ((cosine_sim + 1) / 2) * 100
        return round(similarity_percentage, 2)

    def compare_audio_origin(self):
        """Compara o primeiro ficheiro (referência) com os restantes e apresenta o resultado."""
        if len(self.file_paths) < 2:
            messagebox.showwarning("Aviso", "Por favor, selecione pelo menos 2 ficheiros para comparar.")
            return

        ref_path = self.file_paths[0]
        ref_name = os.path.basename(ref_path)

        results = []
        for path in self.file_paths[1:]:
            file_name = os.path.basename(path)
            try:
                score = self.calculate_similarity(ref_path, path)
                
                # Classificação de acordo com o resultado
                if score >= 90:
                    status = "Origem provável: IDÊNTICA / MESMA GRAVAÇÃO"
                elif score >= 75:
                    status = "Origem provável: SEMELHANTE (Mesmo conteúdo com alterações)"
                else:
                    status = "Origem provável: DIFERENTE"

                results.append(f"• {file_name}:\n   Similaridade: {score}%\n   ({status})\n")
            except Exception as e:
                results.append(f"• {file_name}: Erro no cálculo ({e})\n")

        # Exibe os resultados organizados
        msg_body = f"--- FICHEIRO DE REFERÊNCIA ---\n{ref_name}\n\n--- COMPARAÇÃO DE ORIGEM ---\n" + "\n".join(results)
        
        self.lbl_status.config(text="Comparação concluída com sucesso.", fg="green")
        messagebox.showinfo("Resultado da Comparação de Origem", msg_body)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioComparatorApp(root)
    root.mainloop()