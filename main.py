"""Spam Email Classifier - GUI Interface using Tkinter."""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib

# Add src to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model, load_model
from src.predict import predict_email
from src.preprocess import preprocess_email


# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_classifier.pkl")


class SpamClassifierGUI:
    """GUI Application for Spam Email Classification."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Classifier")
        self.root.geometry("700x600")
        self.root.resizable(True, True)

        self.model = None
        self.model_loaded = False

        # Try to load existing model
        self.try_load_model()

        # Build UI
        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Result.TLabel", font=("Arial", 12))
        style.configure("Spam.TLabel", foreground="red", font=("Arial", 12, "bold"))
        style.configure("Ham.TLabel", foreground="green", font=("Arial", 12, "bold"))

    def try_load_model(self):
        """Attempt to load existing model."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                self.model_loaded = True
            except Exception as e:
                self.model_loaded = False

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_label = ttk.Label(
            main_frame,
            text="Spam Email Classifier",
            style="Header.TLabel"
        )
        header_label.pack(pady=(0, 10))

        # Model status
        self.status_frame = ttk.Frame(main_frame)
        self.status_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(
            self.status_frame,
            text=f"Model: {'Loaded' if self.model_loaded else 'Not Loaded'}",
            foreground="green" if self.model_loaded else "red"
        )
        self.status_label.pack(side=tk.LEFT)

        # Train button
        train_btn = ttk.Button(
            self.status_frame,
            text="Train Model",
            command=self.train_model_gui
        )
        train_btn.pack(side=tk.RIGHT)

        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Email Text", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Text input
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=10,
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Classify button
        classify_btn = ttk.Button(
            button_frame,
            text="Classify Email",
            command=self.classify_email
        )
        classify_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_text
        )
        clear_btn.pack(side=tk.LEFT)

        # Sample button
        sample_btn = ttk.Button(
            button_frame,
            text="Load Sample",
            command=self.load_sample
        )
        sample_btn.pack(side=tk.RIGHT)

        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Classification Result", padding="10")
        result_frame.pack(fill=tk.X, pady=10)

        self.result_label = ttk.Label(
            result_frame,
            text="Enter an email and click 'Classify Email'",
            style="Result.TLabel"
        )
        self.result_label.pack()

        self.confidence_label = ttk.Label(
            result_frame,
            text="",
            font=("Arial", 10)
        )
        self.confidence_label.pack()

        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="About", padding="10")
        info_frame.pack(fill=tk.X, pady=10)

        info_text = (
            "Spam Email Classifier uses Machine Learning (Naive Bayes) to classify\n"
            "emails as SPAM or HAM (not spam).\n\n"
            "Project By- Randika Nawarathne \n"
        
        )
        info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 9))
        info_label.pack()

    def train_model_gui(self):
        """Train model with GUI feedback."""
        if not os.path.exists(DATA_PATH):
            messagebox.showerror("Error", f"Dataset not found at {DATA_PATH}")
            return

        # Show progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Training Model")
        progress_window.geometry("300x100")

        ttk.Label(
            progress_window,
            text="Training model...\nPlease wait.",
            font=("Arial", 10)
        ).pack(expand=True)

        progress_window.update()

        try:
            train_model(
                data_path=DATA_PATH,
                model_path=MODEL_PATH,
                model_type='nb'
            )
            self.model = load_model(MODEL_PATH)
            self.model_loaded = True

            self.status_label.config(
                text="Model: Loaded",
                foreground="green"
            )

            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
        finally:
            progress_window.destroy()

    def classify_email(self):
        """Classify the entered email."""
        if not self.model_loaded:
            messagebox.showwarning(
                "Model Not Loaded",
                "Please train the model first by clicking 'Train Model'."
            )
            return

        text = self.text_input.get("1.0", tk.END).strip()

        if not text:
            messagebox.showwarning("Empty Input", "Please enter an email text.")
            return

        try:
            prediction, confidence = predict_email(self.model, text)

            # Update result label
            if prediction == "spam":
                self.result_label.config(
                    text=f"Result: SPAM",
                    style="Spam.TLabel"
                )
            else:
                self.result_label.config(
                    text=f"Result: HAM (Not Spam)",
                    style="Ham.TLabel"
                )

            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%}"
            )

        except Exception as e:
            messagebox.showerror("Classification Error", str(e))

    def clear_text(self):
        """Clear the text input."""
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Enter an email and click 'Classify Email'", style="Result.TLabel")
        self.confidence_label.config(text="")

    def load_sample(self):
        """Load a sample email for testing."""
        samples = [
            ("spam", "URGENT! You've won $1,000,000! Click here to claim your prize now! http://spam.com"),
            ("ham", "Hi, can we schedule a meeting for tomorrow at 3pm to discuss the project?"),
            ("spam", "FREE MONEY! Make $5000 from home today! No experience needed! Act now!"),
            ("ham", "Thanks for your help with the quarterly report. I really appreciate it."),
        ]

        import random
        label, text = random.choice(samples)

        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", text)

        self.result_label.config(text=f"Sample loaded ({label.upper()})", style="Result.TLabel")
        self.confidence_label.config(text="")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SpamClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
