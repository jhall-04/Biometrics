from PIL import Image, ImageTk
from tkinter import ttk
import tkinter as tk
import cv2

from face.recognition.verify import verify_face

class FaceEnrollApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Verification Tester")

        # Setup camera
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # UI layout
        self.label = ttk.Label(root)
        self.label.grid(row=0, column=0, columnspan=2)

        ttk.Label(root, text="User ID:").grid(row=1, column=0)
        self.user_id_entry = ttk.Entry(root)
        self.user_id_entry.grid(row=1, column=1)

        self.capture_button = ttk.Button(root, text="Verify Face", command=self.capture_face)
        self.capture_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.status_label = ttk.Label(root, text="", foreground="blue")
        self.status_label.grid(row=3, column=0, columnspan=2)

        # Start video loop
        self.update_frame()

        # Clean up on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture_face(self):
        user_id = self.user_id_entry.get().strip()
        if not user_id:
            self.status_label.config(text="⚠️ Enter a user ID", foreground="red")
            return

        frame = self.current_frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        success = verify_face(pil_image, user_id)
        if success:
            self.status_label.config(text=f"✅ Verified '{user_id}'", foreground="green")
        else:
            self.status_label.config(text="❌ Verification failed", foreground="red")

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEnrollApp(root)
    root.mainloop()