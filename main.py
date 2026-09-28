import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image
import cv2
import os
import time
import csv
import re
import threading

# Project-specific imports
from detector import YOLODetector
from classifier import EfficientNetSVMClassifier
from utils import crop_characters, draw_results_on_image, draw_label_on_cropped_image
import output_ai

# Setup Paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LoadingSpinner(ctk.CTkFrame):
    """Custom animated loading spinner widget"""
    def __init__(self, master, size=40, color="#1f6aa5", width=4, **kwargs):
        super().__init__(master, width=size, height=size, fg_color="transparent", **kwargs)
        self.size, self.color, self.width = size, color, width
        self.angle, self.is_spinning, self.animation_id = 0, False, None
        
        bg_color = "#2b2b2b" if ctk.get_appearance_mode() == "Dark" else "#dbdbdb"
        self.canvas = Canvas(self, width=size, height=size, bg=bg_color, highlightthickness=0)
        self.canvas.pack(expand=True, fill="both")

    def _animate(self):
        self.canvas.delete("all")
        if not self.is_spinning: return
            
        pad = 4
        self.canvas.create_arc(pad, pad, self.size - pad, self.size - pad, 
                               start=self.angle, extent=120, style="arc", width=self.width, outline=self.color)
        self.angle = (self.angle - 15) % 360
        self.animation_id = self.after(20, self._animate) 

    def start(self):
        if not self.is_spinning:
            self.is_spinning = True
            self.pack(pady=10)
            self._animate()

    def stop(self):
        self.is_spinning = False
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.pack_forget()

class GuqinApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Guqin Score AI Recognition & Auto-Dapu")
        self.geometry("1400x850") 

        # State Variables
        self.filepath = None
        self.orientation = "vertical"
        self.radical_map = {}
        self.yolo_detector = None
        self.char_classifier = None
        self.is_processing = False

        self._create_widgets()
        
        # Async Model Loading
        self.status_label.configure(text="Loading models...")
        self.spinner.start()
        threading.Thread(target=self._load_resources, daemon=True).start()

    def _create_widgets(self):
        # 1. Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar_frame, text="Guqin AI", font=ctk.CTkFont(size=24, weight="bold")).pack(padx=20, pady=(30, 10))

        self.orientation_var = ctk.StringVar(value="vertical")
        ctk.CTkLabel(self.sidebar_frame, text="Layout Orientation:", anchor="w", font=ctk.CTkFont(weight="bold")).pack(padx=20, pady=(20, 5), anchor="w")
        ctk.CTkRadioButton(self.sidebar_frame, text="Vertical", variable=self.orientation_var, value="vertical", command=self._update_orientation).pack(padx=20, pady=5, anchor="w")
        ctk.CTkRadioButton(self.sidebar_frame, text="Horizontal", variable=self.orientation_var, value="horizontal", command=self._update_orientation).pack(padx=20, pady=5, anchor="w")

        self.select_btn = ctk.CTkButton(self.sidebar_frame, text="Select Image", command=self._select_image)
        self.select_btn.pack(padx=20, pady=(30, 15))

        self.run_btn = ctk.CTkButton(self.sidebar_frame, text="Start Recognition", fg_color="green", state="disabled", command=self._start_process)
        self.run_btn.pack(padx=20, pady=10)

        self.spinner = LoadingSpinner(self.sidebar_frame, size=30, color="#2cc985") 
        ctk.CTkLabel(self.sidebar_frame, text="").pack(fill="y", expand=True)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Initializing...", text_color="gray", wraplength=180, justify="left")
        self.status_label.pack(padx=20, pady=(0, 20))

        # 2. Main Area
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Left: Preview
        self.left_frame = ctk.CTkFrame(self.main_container, corner_radius=10)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.left_frame, text="Original Image", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        self.preview_scroll = ctk.CTkScrollableFrame(self.left_frame)
        self.preview_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_scroll, text="Please select an image", text_color="gray")
        self.preview_label.pack(expand=True, pady=50)

        # Right: Result
        self.right_frame = ctk.CTkFrame(self.main_container, corner_radius=10)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self.right_frame, text="Detection Result", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        self.result_scroll = ctk.CTkScrollableFrame(self.right_frame)
        self.result_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        self.result_label = ctk.CTkLabel(self.result_scroll, text="Results will appear here", text_color="gray")
        self.result_label.pack(expand=True, pady=50)

    def _load_resources(self):
        try:
            self.yolo_detector = YOLODetector(model_path=os.path.join(MODELS_DIR, "yolo11_model.pt"))
            self.char_classifier = EfficientNetSVMClassifier(
                backbone_path=os.path.join(MODELS_DIR, "efficientnet_b0_model.pth"),
                svm_path=os.path.join(MODELS_DIR, "efficientnet_b0_svm_model.pkl"),
                num_classes=203
            )
            self.radical_map = self._load_radical_map(os.path.join(DATA_DIR, 'data.csv'))
            self.status_label.configure(text="Ready.\nPlease select an image.")
        except Exception as e:
            self.status_label.configure(text=f"Init Error:\n{str(e)[:50]}...")
            print(f"Load Error: {e}")
        finally:
            self.spinner.stop()

    def _load_radical_map(self, csv_path):
        mapping = {}
        if not os.path.exists(csv_path): return mapping
        for enc in ['cp932', 'utf-8', 'utf-8-sig']:
            try:
                with open(csv_path, 'r', encoding=enc) as f:
                    for row in csv.DictReader(f):
                        if 'id' in row and 'radical_name' in row: mapping[row['id']] = row['radical_name']
                break
            except: continue
        return mapping

    def _update_orientation(self):
        self.orientation = self.orientation_var.get()

    def _select_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath: return

        self.filepath = filepath
        self._display_image(filepath, self.preview_label)
        self.result_label.configure(image=None, text="Results will appear here")
        self.run_btn.configure(state="normal")
        self.status_label.configure(text=f"Selected:\n{os.path.basename(filepath)}")

    def _display_image(self, img_source, target_label):
        try:
            pil_img = Image.open(img_source) if isinstance(img_source, str) else Image.fromarray(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB))
            base_w = 500
            h_size = int((float(pil_img.size[1]) * float((base_w / float(pil_img.size[0])))))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(base_w, h_size))
            target_label.configure(text="", image=ctk_img)
            target_label.image = ctk_img
        except Exception:
            target_label.configure(text="Display Error")

    def _start_process(self):
        if self.is_processing: return
        self.is_processing = True
        self.run_btn.configure(state="disabled", text="Processing...")
        self.status_label.configure(text="AI is analyzing...")
        self.spinner.start()
        threading.Thread(target=self._process_logic, daemon=True).start()

    def _process_logic(self):
        try:
            sorted_boxes, original_img = self.yolo_detector.detect_and_sort(self.filepath, self.orientation)
            if not sorted_boxes:
                self._update_status_ui("Complete", "No characters found.")
                return

            cropped_imgs = crop_characters(original_img, sorted_boxes)
            predictions = self.char_classifier.classify_all(cropped_imgs, threshold=0.05)

            display_preds, valid_ids = [], []
            for class_id, conf in predictions:
                name = self.radical_map.get(class_id, "")
                if class_id != "Unknown": valid_ids.append(class_id)
                display_preds.append((f"{class_id}\n{name}" if name and class_id != "Unknown" else class_id, conf))

            result_img = draw_results_on_image(original_img, sorted_boxes, display_preds)
            self._save_results(cropped_imgs, display_preds, result_img)

            self.after(0, lambda: self._on_process_complete(result_img, valid_ids))
        except Exception as e:
            self._update_status_ui("Error", f"Failed:\n{e}")
        finally:
            self.is_processing = False

    def _on_process_complete(self, result_image_cv, valid_ids):
        self.spinner.stop()
        self.run_btn.configure(state="normal", text="Start Recognition")
        self.status_label.configure(text="Process Complete!")
        self._display_image(result_image_cv, self.result_label)
        
        if valid_ids:
            self.after(500, lambda: self._ask_music_generation(valid_ids))

    def _update_status_ui(self, title, msg):
        self.after(0, lambda: [
            self.spinner.stop(),
            self.run_btn.configure(state="normal", text="Start Recognition"),
            self.status_label.configure(text=f"{title}\n{msg}")
        ])

    def _save_results(self, cropped, preds, result_img):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        result_dir = os.path.join(OUTPUT_DIR, f"result_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(result_dir, exist_ok=True)

        cv2.imencode(".png", result_img)[1].tofile(os.path.join(result_dir, "full_result.png"))
        for i, (img, pred) in enumerate(zip(cropped, preds)):
            labeled_img = draw_label_on_cropped_image(img, pred)
            safe_name = re.sub(r'[\\/:*?"<>|]', '_', pred[0].replace('\n', '_'))
            cv2.imencode(".png", labeled_img)[1].tofile(os.path.join(result_dir, f"{i+1:03d}_{safe_name}.png"))

    def _ask_music_generation(self, valid_ids):
        if messagebox.askyesno("Generate Music", "Do you want to generate MIDI using AI from the recognized results?"):
            MusicGenDialog(self, valid_ids)

class MusicGenDialog(ctk.CTkToplevel):
    def __init__(self, parent, jzp_ids):
        super().__init__(parent)
        self.title("AI Auto-Dapu Configuration")
        self.geometry("500x550")
        self.jzp_ids = jzp_ids
        self.transient(parent)
        self.grab_set()
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="AI Auto-Dapu Setup", font=("", 18, "bold")).pack(pady=20)
        ctk.CTkLabel(self, text=f"Detected Notes: {len(self.jzp_ids)}").pack()

        frame = ctk.CTkFrame(self)
        frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(frame, text="Song Title (Required):").pack(anchor="w", padx=10, pady=(10, 0))
        self.title_entry = ctk.CTkEntry(frame, placeholder_text="e.g. Shen Ren Chang")
        self.title_entry.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(frame, text="Lyrics / Scene context (Optional):").pack(anchor="w", padx=10)
        self.lyrics_text = ctk.CTkTextbox(frame, height=100)
        self.lyrics_text.pack(fill="x", padx=10, pady=(0, 10))

        self.gen_btn = ctk.CTkButton(self, text="Generate & Save MIDI", command=self._generate, fg_color="purple", height=40)
        self.gen_btn.pack(pady=(20, 10), padx=40, fill="x")

        self.spinner = LoadingSpinner(self, size=40, color="#800080")

    def _generate(self):
        title = self.title_entry.get().strip()
        if not title: return messagebox.showwarning("Input Error", "Please enter a song title.")

        self.gen_btn.configure(state="disabled", text="Generating...")
        self.spinner.start()
        threading.Thread(target=self._generate_thread, args=(title, self.lyrics_text.get("1.0", "end").strip()), daemon=True).start()

    def _generate_thread(self, title, lyrics):
        try:
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
            out_midi = os.path.join(OUTPUT_DIR, f"{safe_title}.mid")

            desc = output_ai.generate_midi_file(
                self.jzp_ids, title, lyrics, out_midi, 
                os.path.join(DATA_DIR, 'data.csv'), os.path.join(DATA_DIR, 'tuning_map.json')
            )
            self.after(0, lambda: self._on_finish(desc, out_midi))
        except Exception as e:
            self.after(0, lambda: self._on_error(e))

    def _on_finish(self, desc, out_midi):
        self.spinner.stop()
        messagebox.showinfo("Success", f"MIDI saved!\n\nAI Concept:\n{desc}\n\nPath: {out_midi}")
        output_ai.open_score_viewer(out_midi)
        self.destroy()

    def _on_error(self, e):
        self.spinner.stop()
        self.gen_btn.configure(state="normal", text="Generate & Save MIDI")
        messagebox.showerror("Error", f"Failed:\n{e}")

if __name__ == '__main__':
    app = GuqinApp()
    app.mainloop()