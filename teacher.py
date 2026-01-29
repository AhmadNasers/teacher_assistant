import os
import base64
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional, List
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime
import pytz
import cv2


class chatState(TypedDict, total=False):
    question : Optional[str]
    summary: Optional[str]
    conversation : List[str]
    last_user_message: Optional[str]
    is_question_exist: Optional[bool]

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

def greeting(state: chatState) -> chatState:
    greet_message = "hello how can i help you"
    print(greet_message)
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"messages": [], "question": ""}
    
    if "messages" not in data:
        data["messages"] = []
    
    data["messages"].append({"sender": "chatbot", "message": greet_message})
    
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return state


def capture_question(state: chatState) -> chatState:
    print("Opening camera... Press 's' to capture the question.")
    
    # Ensure the directory exists
    os.makedirs("question", exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        state['is_question_exist'] = False
        return state
    
    # Create window explicitly
    window_name = "Capture Question (Press 's' to save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Capturing...")
            # Visual feedback: show the captured frame for a moment
            cv2.imshow(window_name, frame)
            cv2.waitKey(1000)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join("question", f"img_{timestamp}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Image saved at {file_path}")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return state


def convert_to_text(state: chatState) -> chatState:
    print("Converting image to text...")
    
    question_dir = "question"
    if not os.path.exists(question_dir):
        print("No question folder found")
        return state
    
    files = [os.path.join(question_dir, f) for f in os.listdir(question_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        print("No images found in question folder")
        return state
        
    latest_file = max(files, key=os.path.getctime)
    print(f"Processing file: {latest_file}")
    
    with open(latest_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract the text from this image exactly as it appears."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
        ]
    )
    
    try:
        response = llm.invoke([message])
        extracted_text = response.content
        print(f"Extracted Text: {extracted_text}")
        
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                content = f.read()
                data = json.loads(content) if content else {"messages": [], "question": ""}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"messages": [], "question": ""}
            
        if not isinstance(data, dict):
             data = {"messages": [], "question": ""}
             
        data["question"] = extracted_text
        
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
        state['question'] = extracted_text
        state['is_question_exist'] = True
        
    except Exception:
        pass
        
    return state

import sys
# ... (GUI Imports) ...


import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                             QLabel, QStackedWidget, QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import re

def get_question_image_path():
    question_dir = "question"
    if not os.path.exists(question_dir): return None
    files = [os.path.join(question_dir, f) for f in os.listdir(question_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: return None
    return max(files, key=os.path.getctime)

def cleanup_data(state: dict = None) -> dict:
    print("Cleaning up data for restart...")
    question_dir = "question"
    if os.path.exists(question_dir):
        for f in os.listdir(question_dir):
            file_path = os.path.join(question_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    data = {"messages": [], "question": ""}
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return {}

def check_question_exists(state: dict = None) -> str:
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            content = f.read()
            if not content: return "capture"
            data = json.loads(content)
            if data.get("question"):
                return "chat"
    except Exception:
        pass
    return "capture"


# Worker thread for long-running AI tasks to keep UI responsive
class AIWorker(QThread):
    response_ready = pyqtSignal(str, str) # sender, message

    def __init__(self, task_type, payload):
        super().__init__()
        self.task_type = task_type
        self.payload = payload

    def run(self):
        timezone = pytz.timezone('Africa/Cairo')
        
        if self.task_type == "convert_and_greet":
             # 1. Convert
            dummy_state = {}
            # Reusing code from convert_to_text but modified to not depend on state return
            img_path = get_question_image_path()
            if img_path:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                message = HumanMessage(content=[
                    {"type": "text", "text": "Extract the text from this image exactly as it appears. If no text is found or image is unclear, reply exactly with 'EXTRACTION_FAILED'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                ])
                try:
                    response = llm.invoke([message])
                    extracted_text = response.content.strip()
                    
                    if extracted_text == "EXTRACTION_FAILED":
                        self.response_ready.emit("chatbot", "I couldn't read the question from the image. Please retake the photo.")
                        self.response_ready.emit("system", "RETAKE_PHOTO") # Signal for UI to switch back
                        return

                    # Update JSON
                    try:
                        with open("data.json", "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except: data = {"messages": [], "question": ""}
                    if "messages" not in data: data["messages"] = []
                    
                    data["question"] = extracted_text
                    
                    # Send extracted question to student
                    msg_question = f"I read this question: \"{extracted_text}\""
                    data["messages"].append({
                        "sender": "chatbot", 
                        "message": msg_question,
                        "timestamp": datetime.now(timezone).isoformat()
                    })
                    self.response_ready.emit("chatbot", msg_question)
                    
                    # Add Greeting
                    greet_message = "hello how can i help you"
                    data["messages"].append({
                        "sender": "chatbot", 
                        "message": greet_message,
                        "timestamp": datetime.now(timezone).isoformat()
                    })
                    
                    with open("data.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                        
                    self.response_ready.emit("chatbot", greet_message)
                except Exception as e:
                    self.response_ready.emit("chatbot", f"Error processing image: {str(e)}")
            else:
                self.response_ready.emit("chatbot", "Error: No image found.")

        elif self.task_type == "chat":
            user_msg = self.payload['message']
            # Load context
            try:
                with open("data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
            except: data = {"messages": [], "question": ""}
            
            system_prompt = r"""You are an expert math teacher. Your role is to:
            - Guide students to think and find the solution on their own.
            - Provide step-by-step support without solving the problem directly.
            - Be patient and encouraging.
            - If a student is struggling significantly, you can provide more direct help and eventually the solution.
            
            CRITICAL FORMATTING RULES - Follow these EXACTLY:
            - For mathematical expressions, ONLY use dollar signs: $expression$ for inline math
            - For display equations, ONLY use: $$expression$$ for block math  
            - NEVER use \( \) or \[ \] - these cause display errors
            - NEVER use backslashes before parentheses in math
            - Example: Write $27y^2$ NOT \(27y^2\)
            - Example: Write $$x^2 + y^2 = r^2$$ NOT \[x^2 + y^2 = r^2\]
            
            If there's a question stored in the context, you can reference it in your responses."""
            
            history = [SystemMessage(content=system_prompt)]
            
            # Add existing question context if available
            if data.get("question"):
                history.append(SystemMessage(content=f"The student is asking about this question: {data['question']}"))
            
            # Context history
            visible_msgs = data.get("messages", [])[-10:] # last 10
            for m in visible_msgs:
                if m['sender'] == 'student':
                    history.append(HumanMessage(content=m['message']))
                elif m['sender'] == 'chatbot':
                    history.append(AIMessage(content=m['message']))
            
            # Append current
            history.append(HumanMessage(content=user_msg))
            
            try:
                response = llm.invoke(history)
                ai_text = response.content
                
                # Save to JSON
                if "messages" not in data: list(data["messages"])
                
                # Note: User message already saved by UI before calling thread?
                # Better to save here to ensure order or save in UI.
                # UI saves user msg. We handle AI msg.
                
                timestamp = datetime.now(timezone).isoformat()
                chatbot_msg = {
                    "sender": "chatbot",
                    "message": ai_text,
                    "timestamp": timestamp
                }
                data["messages"].append(chatbot_msg)
                
                with open("data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    
                self.response_ready.emit("chatbot", ai_text)
                
            except Exception as e:
                self.response_ready.emit("chatbot", f"Error: {e}")

class TeacherGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Teacher")
        self.resize(1000, 800)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        
        # Page 1: Capture
        self.capture_page = QWidget()
        self.setup_capture_page()
        self.stacked_widget.addWidget(self.capture_page)
        
        # Page 2: Chat
        self.chat_page = QWidget()
        self.setup_chat_page()
        self.stacked_widget.addWidget(self.chat_page)
        
        # Logic
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.cap = None
        
        # Check start state
        self.check_initial_state()

    def setup_capture_page(self):
        layout = QVBoxLayout(self.capture_page)
        
        self.lbl_camera = QLabel("Camera initialization...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.lbl_camera.setMinimumSize(640, 480)
        layout.addWidget(self.lbl_camera)
        
        btn_capture = QPushButton("Capture Question")
        btn_capture.setFixedHeight(50)
        btn_capture.setStyleSheet("""
            QPushButton {
                background-color: #007acc; 
                color: white; 
                font-size: 16px; 
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #005999; }
        """)
        btn_capture.clicked.connect(self.take_photo)
        layout.addWidget(btn_capture)

    def setup_chat_page(self):
        layout = QVBoxLayout(self.chat_page)
        
        # Header with Restart
        header = QHBoxLayout()
        title = QLabel("Chat Session")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        
        btn_restart = QPushButton("Restart Session")
        btn_restart.setStyleSheet("""
            background-color: #d9534f; color: white; border: none; padding: 5px 15px; border-radius: 4px;
        """)
        btn_restart.clicked.connect(self.restart_session)
        header.addWidget(btn_restart)
        layout.addLayout(header)
        
        # Chat Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch() # Push messages down
        self.scroll_area.setWidget(self.chat_container)
        self.scroll_area.setStyleSheet("border: none; background-color: #252526;")
        layout.addWidget(self.scroll_area)
        
        # Input Area
        input_layout = QHBoxLayout()
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Type your message...")
        self.txt_input.setStyleSheet("padding: 10px; border-radius: 4px; background-color: #3c3c3c; color: white;")
        self.txt_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.txt_input)
        
        btn_send = QPushButton("Send")
        btn_send.setStyleSheet("background-color: #28a745; color: white; padding: 10px 20px; border-radius: 4px;")
        btn_send.clicked.connect(self.send_message)
        input_layout.addWidget(btn_send)
        
        layout.addLayout(input_layout)

    def check_initial_state(self):
        # reuse check_question_exists logic
        state = check_question_exists({}) # check file
        if state == "chat":
            self.load_history()
            self.stacked_widget.setCurrentWidget(self.chat_page)
        else:
            self.start_camera()
            self.stacked_widget.setCurrentWidget(self.capture_page)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30) # 30ms

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None # Ensure it's reset

    def update_camera(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                # Convert to Qt format
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.lbl_camera.setPixmap(QPixmap.fromImage(p))
                self.current_frame = frame

    def take_photo(self):
        self.stop_camera()
        
        # Save image
        os.makedirs("question", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("question", f"img_{timestamp}.jpg")
        if hasattr(self, 'current_frame'):
            cv2.imwrite(file_path, self.current_frame)
            print(f"Captured {file_path}")
            
            # Start AI processing
            # Show loading?
            self.add_message("system", "Processing image and generating greeting...")
            self.stacked_widget.setCurrentWidget(self.chat_page)
            
            self.worker = AIWorker("convert_and_greet", {})
            self.worker.response_ready.connect(self.handle_ai_response)
            self.worker.start()

    def handle_ai_response(self, sender, message):
        if sender == "system" and message == "RETAKE_PHOTO":
            QTimer.singleShot(2000, lambda: self.stacked_widget.setCurrentWidget(self.capture_page))
            QTimer.singleShot(2000, self.start_camera)
            return
            
        # Animate only chatbot messages
        animate = (sender == "chatbot")
        self.add_message(sender, message, animate=animate)

    def send_message(self):
        text = self.txt_input.text().strip()
        if not text: return
        
        self.txt_input.clear()
        self.add_message("student", text)
        
        timezone = pytz.timezone('Africa/Cairo')
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except: data = {"messages": [], "question": ""}
        if "messages" not in data: data["messages"] = []
        
        data["messages"].append({
            "sender": "student", 
            "message": text, 
            "timestamp": datetime.now(timezone).isoformat()
        })
        with open("data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        self.worker = AIWorker("chat", {"message": text})
        self.worker.response_ready.connect(self.handle_ai_response)
        self.worker.start()

    def add_message(self, sender, text, animate=False):
        # Use a container that can hold multiple widgets (text and math)
        message_container = QWidget()
        message_layout = QVBoxLayout(message_container)
        message_layout.setContentsMargins(10, 10, 10, 10)
        message_layout.setSpacing(5) # Space between text/math blocks

        # Parse the text for LaTeX - handle both valid and malformed expressions
        # First, clean up malformed LaTeX patterns before parsing
        text = re.sub(r'\\\(([^\\]*?)\\\)', r'$\1$', text)  # Convert \(...\) to $...$, allowing parentheses inside
        text = re.sub(r'\\\(([^\\]*?)\\?$', r'$\1$', text)  # Handle incomplete \(...
        text = re.sub(r'\\\[([^\\]*?)\\\]', r'$$\1$$', text)  # Convert \[...\] to $$...$$
        text = re.sub(r'\\\[([^\\]*?)\\?$', r'$$\1$$', text)  # Handle incomplete \[...
        
        parts = re.split(r'(\$\$[^\$]*\$\$|\$[^\$]*\$)', text)

        for part in parts:
            if not part:
                continue
            
            widget = None
            if part.startswith('$$') and part.endswith('$$'):
                # Block LaTeX
                latex = part[2:-2].strip()
                pixmap = self.render_latex(latex, font_size=14, is_block=True)
                if pixmap:
                    lbl = QLabel()
                    lbl.setPixmap(pixmap)
                    lbl.setAlignment(Qt.AlignCenter)
                    widget = lbl
                else:
                    # Fallback to plain text
                    lbl = QLabel(part)
                    lbl.setWordWrap(True)
                    lbl.setFont(QFont("Arial", 12))
                    widget = lbl
            elif part.startswith('$') and part.endswith('$'):
                # Inline LaTeX
                latex = part[1:-1].strip()
                if latex:  # Only render if there's actual content
                    pixmap = self.render_latex(latex, font_size=12, is_block=False)
                    if pixmap:
                        lbl = QLabel()
                        lbl.setPixmap(pixmap)
                        lbl.setAlignment(Qt.AlignLeft)
                        widget = lbl
                    else:
                        # Fallback to plain text
                        lbl = QLabel(f"${latex}$")
                        lbl.setWordWrap(True)
                        lbl.setFont(QFont("Arial", 12))
                        widget = lbl

            else:
                # Plain text
                lbl = QLabel(part)
                lbl.setWordWrap(True)
                lbl.setFont(QFont("Arial", 12))
                lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                widget = lbl

            if widget:
                message_layout.addWidget(widget)

        # Styling and layout for the whole message bubble
        bubble = QFrame()
        bubble.setLayout(message_layout)
        bubble.setMaximumWidth(int(self.scroll_area.width() * 0.75))

        container = QFrame()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0,0,0,0)

        if sender == "student":
            bubble.setStyleSheet("background-color: #007acc; color: white; padding: 10px; border-radius: 10px;")
            layout.addStretch()
            layout.addWidget(bubble)
        elif sender == "chatbot":
            bubble.setStyleSheet("background-color: #3a3a3a; color: white; padding: 10px; border-radius: 10px;")
            layout.addWidget(bubble)
            layout.addStretch()
        else: # system
            bubble.setStyleSheet("color: #888; font-style: italic;")
            layout.addWidget(bubble)
            layout.setAlignment(Qt.AlignCenter)

        self.chat_layout.addWidget(container)
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
        
        if animate and sender == "chatbot":
            # For animated text, we need to handle LaTeX differently
            # For simplicity, we'll disable animation for LaTeX content
            if '$' not in text:
                self.animate_text_simple(message_layout, text)

    def render_latex(self, latex_string, font_size=12, is_block=False):
        try:
            # Set matplotlib to use non-interactive backend
            plt.rcParams['text.usetex'] = False
            
            fig, ax = plt.subplots(figsize=(0.1, 0.1))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.axis('off')
            
            # Render the LaTeX
            text_obj = ax.text(0.5, 0.5, f'${latex_string}$', 
                             fontsize=font_size, 
                             color='white',
                             ha='center', 
                             va='center',
                             transform=ax.transAxes)
            
            # Get the bounding box
            fig.canvas.draw()
            bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            
            # Convert to display coordinates
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            
            # Adjust figure size
            width = bbox.width + 0.1
            height = bbox.height + 0.1
            fig.set_size_inches(width, height)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, 
                       bbox_inches='tight', pad_inches=0.05, dpi=150)
            plt.close(fig)
            buf.seek(0)
            
            # Convert to QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            return pixmap
            
        except Exception as e:
            print(f"Failed to render LaTeX '{latex_string}': {e}")
            return None

    def animate_text(self, label, full_text):
        label.current_text = ""
        label.full_text = full_text
        label.char_index = 0
        
        def step():
            if label.char_index < len(label.full_text):
                # Add a few chars at a time for speed (otherwise long msgs take forever)
                chunk_size = 2 
                chunk = label.full_text[label.char_index : label.char_index + chunk_size]
                label.current_text += chunk
                label.setText(label.current_text)
                label.char_index += chunk_size
                
                # Auto-scroll
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                )
            else:
                label.typing_timer.stop()
                
        label.typing_timer = QTimer(self)
        label.typing_timer.timeout.connect(step)
        label.typing_timer.start(15) # 15ms delay

    def animate_text_simple(self, layout, full_text):
        # Simple animation for plain text without LaTeX
        # Find the first QLabel in the layout to animate
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                self.animate_text(widget, full_text)
                break

    def load_history(self):
        # Clear existing
        # Iterate in reverse to safely remove widgets
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        # Re-add the stretch item if it was removed
        self.chat_layout.addStretch()
            
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                for msg in data.get("messages", []):
                    self.add_message(msg["sender"], msg["message"])
        except: pass

    def restart_session(self):
        # Cleanup
        cleanup_data()
        
        # Clear chat UI
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        self.chat_layout.addStretch() # Re-add stretch
            
        # Switch to capture
        self.start_camera()
        self.stacked_widget.setCurrentWidget(self.capture_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TeacherGUI()
    window.show()
    sys.exit(app.exec_())

