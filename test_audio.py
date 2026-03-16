import soundcard as sc
import numpy as np
import time
import threading

def test_device(mic, duration=5.0):
    try:
        with mic.recorder(samplerate=16000, channels=2) as recorder:
            frames = []
            start_time = time.time()
            while time.time() - start_time < duration:
                data = recorder.record(numframes=1024)
                frames.append(data)
            
            if frames:
                all_data = np.concatenate(frames)
                max_val = np.max(np.abs(all_data))
                with open("results.txt", "a", encoding="utf-8") as f: f.write(f"[OK] [Glosnosc: {max_val:.4f}] - {mic.name}\n")
            else:
                with open("results.txt", "a", encoding="utf-8") as f: f.write(f"[PUSTO] - {mic.name}\n")
    except Exception as e:
        with open("results.txt", "a", encoding="utf-8") as f: f.write(f"[BLAD] - {mic.name}: {e}\n")

if __name__ == "__main__":
    with open("results.txt", "w", encoding="utf-8") as f: f.write("START TESTU\n")
    print("Rozpoczynam testowanie urządzeń audio (5 sekund nasłuchiwania)...")
    print("Włącz proszę na ten czas jakiś dźwięk na komputerze (np. filmik na YT lub kogoś na Discordzie)!\n")
    
    mics = sc.all_microphones(include_loopback=True)
    threads = []
    
    for mic in mics:
        t = threading.Thread(target=test_device, args=(mic, 5.0))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("\nTest zakończony.")
