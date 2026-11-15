"""
Erzeugt die Bildtextur textures/marmor.jpg (1024 × 512, nahtlos kachelbar).

Marmor in Türkis und Dunkelblau mit hellen und goldenen Adern.
Grundlage ist periodisches Rauschen: Zufällige Frequenzen werden im Fourier-Raum
gefiltert und zurücktransformiert – das Ergebnis wiederholt sich automatisch an
allen Rändern, sodass auf dem Torus keine Naht entsteht.

Aufruf:  python3 marmor.py   (benötigt numpy und Pillow)
"""
import numpy as np
from PIL import Image

W, H = 1024, 512
rng = np.random.default_rng(20261115)


def periodic_noise(octave_falloff=1.9, low=1, high=180):
    """Fraktales, periodisches Rauschen über ein 1/f-Spektrum, Werte in [-1, 1]."""
    fy = np.fft.fftfreq(H) * H
    fx = np.fft.rfftfreq(W) * W
    f = np.sqrt(fx[None, :] ** 2 + (fy[:, None] * 2) ** 2)   # Pixel sind 2:1 verteilt → isotrop machen
    amp = np.where((f >= low) & (f <= high), 1.0 / np.maximum(f, 1) ** octave_falloff, 0)
    spec = amp * np.exp(2j * np.pi * rng.random(amp.shape))
    n = np.fft.irfft2(spec, s=(H, W))
    return n / np.abs(n).max()


x = np.linspace(0, 1, W, endpoint=False)[None, :]
y = np.linspace(0, 1, H, endpoint=False)[:, None]

turb1 = periodic_noise(2.2, high=40)          # weich: verwirbelt die Adern
turb2 = periodic_noise(1.8, low=2, high=90)
fine = periodic_noise(1.1, low=20, high=400)

# Hauptadern: Sinus über eine schräge, ganzzahlig periodische Richtung, durch Rauschen verwirbelt
phase = 2 * np.pi * (3 * x + 2 * y) + 5.5 * turb1 + 2.0 * turb2
veins = np.abs(np.sin(phase))                 # 0 auf der Ader, 1 dazwischen
vein_white = np.exp(-(veins / 0.035) ** 2) + 0.35 * np.exp(-(veins / 0.16) ** 2)   # Kern + Schimmer
phase2 = 2 * np.pi * (-2 * x + 3 * y) + 4.0 * turb1 + 2.5 * turb2
vein_gold = np.exp(-(np.abs(np.sin(phase2)) / 0.025) ** 2) * np.clip(0.5 + turb2, 0, 1)

# Grundfarbe: Verlauf von Dunkelblau über Petrol nach Türkis
base = 0.5 + 0.5 * (0.7 * turb2 + 0.3 * fine)
base = np.clip((base - 0.2) / 0.6, 0, 1)[..., None]
navy = np.array([0.04, 0.09, 0.22])
petrol = np.array([0.03, 0.36, 0.45])
turq = np.array([0.30, 0.82, 0.80])
col = np.where(base < 0.5, navy + (petrol - navy) * (base / 0.5), petrol + (turq - petrol) * ((base - 0.5) / 0.5))

# Wolkige Einschlüsse und feine Körnung
col = col * (0.88 + 0.12 * fine[..., None])
col = col + (np.array([0.85, 0.97, 0.96]) - col) * np.clip(0.85 * vein_white[..., None], 0, 1)
col = col + (np.array([0.95, 0.76, 0.36]) - col) * (0.9 * vein_gold[..., None])

img = (np.clip(col, 0, 1) ** (1 / 1.05) * 255).astype(np.uint8)
Image.fromarray(img).save("marmor.jpg", quality=90, optimize=True, progressive=True)
print("marmor.jpg gespeichert:", img.shape)
