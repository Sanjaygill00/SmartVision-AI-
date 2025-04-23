import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
from collections import Counter

model = YOLO('yolov8n.pt')

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        detect_objects(file_path)

def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    result = results[0]
    annotated_image = result.plot()
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize((450, 450))
    image_tk = ImageTk.PhotoImage(image_pil)

    # Show image
    image_label.config(image=image_tk)
    image_label.image = image_tk

    # Clear previous results
    for widget in result_frame.winfo_children():
        widget.destroy()

    # Extract details
    height, width = image.shape[:2]
    class_ids = result.boxes.cls.tolist()
    class_names = [model.names[int(cls)] for cls in class_ids]
    counts = Counter(class_names)
    speed = result.speed
    input_shape = (1, 3, height, width)

    current_row = 0

    # Image Size
    add_label(result_frame, "📐 Image Size", current_row, "#00BCD4", True)
    current_row += 1
    add_label(result_frame, f"{height} height x {width} width", current_row)
    current_row += 1

    # Detected Objects
    add_label(result_frame, "📦 Detected Objects", current_row, "#4CAF50", True)
    current_row += 1
    if counts:
        for obj, count in counts.items():
            add_label(result_frame, f"{obj.capitalize()}:", current_row, "#E8F5E9", False, 1)
            add_label(result_frame, f"{count}", current_row, "#FFF3E0", False, 2)
            current_row += 1
    else:
        add_label(result_frame, "No objects detected", current_row)
        current_row += 1

    # Detection Time
    add_label(result_frame, "⏱️ Detection Time", current_row, "#FFC107", True)
    current_row += 1
    add_label(result_frame, "Preprocess:", current_row, "#FFF8E1", False, 1)
    add_label(result_frame, f"{speed['preprocess']:.1f} ms", current_row, "#FFF3E0", False, 2)
    current_row += 1
    add_label(result_frame, "Inference:", current_row, "#FFF8E1", False, 1)
    add_label(result_frame, f"{speed['inference']:.1f} ms", current_row, "#FFF3E0", False, 2)
    current_row += 1
    add_label(result_frame, "Postprocess:", current_row, "#FFF8E1", False, 1)
    add_label(result_frame, f"{speed['postprocess']:.1f} ms", current_row, "#FFF3E0", False, 2)
    current_row += 1

    # Inference Shape
    add_label(result_frame, "📊 Inference Shape", current_row, "#3F51B5", True)
    current_row += 1
    add_label(result_frame, str(input_shape), current_row)
    current_row += 1

    # YOLO Summary
    add_label(result_frame, "📋 YOLO Summary", current_row, "#9C27B0", True)
    current_row += 1
    summary = f"0: {height}x{width} " + ", ".join([f"{count} {obj}" for obj, count in counts.items()]) + f", {speed['inference']:.1f}ms"
    add_label(result_frame, summary, current_row)

def add_label(frame, text, row, bg="#21222c", bold=False, col=0):
    # Choose contrasting text color based on background
    dark_text = "#21222c"  # Very dark for light backgrounds
    light_text = "white"   # White for dark backgrounds
    text_color = dark_text if bg in ["#FFF3E0", "#FFF8E1", "#E8F5E9"] else light_text

    font = ("Consolas", 11, "bold" if bold else "normal")
    lbl = tk.Label(frame, text=text, fg=text_color, bg=bg, anchor="w", font=font, justify="left")
    lbl.grid(row=row, column=col, sticky="w", padx=10, pady=4)


# GUI Setup
root = tk.Tk()
root.title("YOLOv8 Object Detection Viewer")
root.geometry("1200x600")
root.config(bg="#1e1e2f")

# Upload Button
upload_btn = tk.Button(root, text="Upload Image", font=("Helvetica", 14, "bold"),
                       bg="#3A7AFE", fg="white", command=select_image)
upload_btn.pack(pady=10)

# Main layout container
main_frame = tk.Frame(root, bg="#1e1e2f")
main_frame.pack(fill="both", expand=True)

# Left: Image display
image_label = tk.Label(main_frame, bg="#1e1e2f")
image_label.pack(side="left", padx=30, pady=10)  # <-- Left margin added here

# Right: Scrollable results
right_frame = tk.Frame(main_frame, bg="#1e1e2f")
right_frame.pack(side="right", fill="both", expand=True)

canvas = tk.Canvas(right_frame, bg="#1e1e2f", highlightthickness=0)
scroll_y = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas, bg="#1e1e2f")

canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
canvas.configure(yscrollcommand=scroll_y.set)
scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

result_frame = tk.Frame(scroll_frame, bg="#1e1e2f")
result_frame.pack(fill="both", expand=True)

# Footer
footer = tk.Label(root, text="Made with ❤️ using YOLOv8 + Tkinter",
                  bg="#1e1e2f", fg="gray", font=("Helvetica", 10))
footer.pack(side="bottom", pady=5)

root.mainloop()
