from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return 'No file uploaded.', 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    name, ext = os.path.splitext(filename)
    # The Real-ESRGAN script will create the output with this name
    output_filename = f'{name}_out{ext}'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    command = [
        "python", "Real-ESRGAN/inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_path,
        "-o", OUTPUT_FOLDER,
        "--suffix", "out"  # Use default 'out' suffix as expected by the script
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    print(f"Command output: {result.stdout}")
    print(f"Command error: {result.stderr}")
    
    # Check if output file exists
    if os.path.exists(output_path):
        print(f"Output file exists at: {output_path}")
    else:
        print(f"Output file not found at: {output_path}")
        # List files in OUTPUT_FOLDER to see what was created
        print(f"Files in output folder: {os.listdir(OUTPUT_FOLDER)}")

    return redirect(url_for('show_result', filename=output_filename))

@app.route('/result/<filename>')
def show_result(filename):
    # Check if file exists
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(output_path):
        return f'''
            <h2>Upscaled Result:</h2>
            <img src="/static/outputs/{filename}" style="max-width:500px;">
            <br><br><a href="/">Upload Another</a>
        '''
    else:
        # If not found, list available files
        available_files = os.listdir(OUTPUT_FOLDER)
        return f'''
            <h2>Error: Output file not found</h2>
            <p>Could not find {filename} in output folder.</p>
            <p>Available files: {', '.join(available_files) if available_files else 'None'}</p>
            <br><a href="/">Upload Another</a>
        '''

if __name__ == '__main__':
    app.run(debug=True)
