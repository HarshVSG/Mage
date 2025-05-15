from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import subprocess
import time
import glob
import datetime
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Add a session tracking mechanism to keep track of files
SESSION_FILES = {}

# Available models with scale factors
MODELS = {
    'RealESRGAN_x4plus': {
        'name': 'Standard (x4)',
        'description': 'General purpose model with good balance of detail and speed',
        'default_scale': 4
    },
    'RealESRGAN_x4plus_anime_6B': {
        'name': 'Anime (x4)',
        'description': 'Specialized model for anime and cartoon images',
        'default_scale': 4
    },
    'realesr-general-x4v3': {
        'name': 'Photo Realistic (x4)',
        'description': 'Enhanced detail for photographic images',
        'default_scale': 4
    }
}

# Available resolution options
RESOLUTIONS = {
    '2': 'Double (2x)',
    '4': 'Quadruple (4x)'
}

def cleanup_old_files(max_age_hours=24):
    """Delete files older than max_age_hours in upload and output folders"""
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        files = glob.glob(os.path.join(folder, '*'))
        current_time = time.time()
        
        for f in files:
            try:
                # Get file modification time
                file_time = os.path.getmtime(f)
                # If file is older than max_age_hours
                if (current_time - file_time) > (max_age_hours * 3600):
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                    print(f"Deleted old file: {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")

def cleanup_immediately():
    """Force cleanup of all files in upload and output folders"""
    try:
        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            files = glob.glob(os.path.join(folder, '*'))
            for f in files:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                    print(f"Deleted file: {f}")
                except Exception as e:
                    print(f"Error deleting {f}: {e}")
    except Exception as e:
        print(f"Error during immediate cleanup: {e}")

@app.route('/')
def index():
    # Clean up all files when returning to the homepage
    cleanup_immediately()
    # Clear session tracking
    SESSION_FILES.clear()
    return render_template('index.html', models=MODELS, resolutions=RESOLUTIONS)

@app.route('/upload', methods=['POST'])
def upload():
    # Clean up all files from previous sessions
    cleanup_immediately()
    
    # Check if files are present in the request
    if 'image' not in request.files:
        print("No 'image' in request.files")
        return 'No file uploaded.', 400
        
    file = request.files['image']
    
    # Check if the file has a filename (empty files have no filename)
    if file.filename == '':
        print("Empty filename")
        return 'No file selected.', 400
    
    model = request.form.get('model', 'RealESRGAN_x4plus')
    resolution = request.form.get('resolution', '4')  # Default to 4x if not specified
    denoise = request.form.get('denoise', '0.5')  # Get denoise value
    
    print(f"Processing with model: {model}, resolution: {resolution}, denoise: {denoise}")
    
    if model not in MODELS:
        print(f"Invalid model: {model}")
        return 'Invalid model selected.', 400
        
    if resolution not in RESOLUTIONS:
        print(f"Invalid resolution: {resolution}")
        return 'Invalid resolution selected.', 400
    
    # Fix potential path traversal issues with secure_filename
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Ensure the directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Save the uploaded file
    try:
        file.save(input_path)
        print(f"File saved to {input_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return f'Error saving file: {str(e)}', 500
    
    # Track this file in our session
    session_id = request.cookies.get('session', str(time.time()))
    if session_id not in SESSION_FILES:
        SESSION_FILES[session_id] = []
    SESSION_FILES[session_id].append(input_path)
    
    name, ext = os.path.splitext(filename)
    # The Real-ESRGAN script will create the output with this name pattern by default
    output_filename = f'{name}_out{ext}'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Build the command with the denoise parameter
    command = [
        "python", "Real-ESRGAN/inference_realesrgan.py",
        "-n", model,
        "-i", input_path,
        "-o", OUTPUT_FOLDER,
        "--fp32",  # Use full precision to avoid potential issues
        "-s", resolution  # Use the selected resolution scale factor
    ]
    
    # Add denoise parameter if it's not "0" (None)
    if denoise != "0":
        command.extend(["-d", denoise])
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        # Set a timeout to prevent hanging
        result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)), 
                              capture_output=True, text=True, timeout=120)
        print(f"Command output: {result.stdout}")
        print(f"Command error: {result.stderr}")
        
        # If the subprocess returns non-zero, there was an error
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            return f'Processing failed with error: {result.stderr}', 500
    except subprocess.TimeoutExpired:
        print("Command timed out after 120 seconds")
        return 'Processing timed out, please try again with a smaller image.', 504
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return f'Error processing image: {str(e)}', 500
    
    # Check if output file exists
    if os.path.exists(output_path):
        print(f"Output file exists at: {output_path}")
        # Track this output file in our session
        SESSION_FILES[session_id].append(output_path)
    else:
        print(f"Output file not found at: {output_path}")
        # List files in OUTPUT_FOLDER to see what was created
        print(f"Files in output folder: {os.listdir(OUTPUT_FOLDER)}")
        # Look for any file that might have been created with a similar name
        possible_files = [f for f in os.listdir(OUTPUT_FOLDER) if name in f]
        if possible_files:
            output_filename = possible_files[0]
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            print(f"Found possible output file: {output_path}")
            # Track this output file in our session
            SESSION_FILES[session_id].append(output_path)
        else:
            return 'Processing completed but output file was not created. Please try again.', 500

    return redirect(url_for('show_result', filename=output_filename, original=filename, model=model, resolution=resolution))

@app.route('/result/<filename>')
def show_result(filename):
    # Get original filename and model used
    original = request.args.get('original', '')
    model_id = request.args.get('model', 'RealESRGAN_x4plus')
    resolution = request.args.get('resolution', '4')
    model_name = MODELS.get(model_id, {}).get('name', 'Standard')
    resolution_name = RESOLUTIONS.get(resolution, 'Quadruple (4x)')
    
    # Check if files exist
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    original_path = os.path.join(UPLOAD_FOLDER, original)
    
    if os.path.exists(output_path):
        # Get file sizes
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # in MB
        original_size = os.path.getsize(original_path) / (1024 * 1024) if os.path.exists(original_path) else 0
        
        # Get image dimensions if PIL is available
        original_dimensions = ""
        output_dimensions = ""
        try:
            from PIL import Image
            if os.path.exists(original_path):
                with Image.open(original_path) as img:
                    original_dimensions = f"{img.width} × {img.height}"
            if os.path.exists(output_path):
                with Image.open(output_path) as img:
                    output_dimensions = f"{img.width} × {img.height}"
        except ImportError:
            print("PIL not available for image dimension analysis")
        
        # Add a JavaScript function to clean up when page is closed or navigated away from
        cleanup_script = """
        <script>
        // Clean up on page unload or when navigating away
        window.addEventListener('beforeunload', function() {
            navigator.sendBeacon('/cleanup');
        });
        
        // Also clean up when clicking the "Upscale Another Image" link
        document.addEventListener('DOMContentLoaded', function() {
            const upscaleLink = document.querySelector('a[href="/"]');
            if (upscaleLink) {
                upscaleLink.addEventListener('click', function(e) {
                    // We'll still navigate to the link, but first let's send our cleanup request
                    navigator.sendBeacon('/cleanup');
                });
            }
        });
        </script>
        """
        
        return render_template('result.html', 
                              original_filename=original,
                              output_filename=filename,
                              model_name=model_name,
                              resolution_name=resolution_name)
    else:
        # If not found, list available files
        available_files = os.listdir(OUTPUT_FOLDER)
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Image Not Found</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <style>
                :root {{
                    --primary-color: #4a6cf7;
                    --error-color: #e74c3c;
                    --light-bg: #f8f9ff;
                    --border-radius: 15px;
                    --box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    --transition: all 0.3s ease;
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Poppins', sans-serif;
                }}
                
                body {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 40px 20px;
                }}
                
                .container {{
                    max-width: 800px;
                    width: 100%;
                    background-color: white;
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                    padding: 30px;
                    margin-top: 20px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .container::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 5px;
                    background: linear-gradient(90deg, var(--error-color), #c0392b);
                }}
                
                h2 {{
                    color: var(--error-color);
                    margin-bottom: 20px;
                    font-weight: 600;
                }}
                
                p {{
                    margin-bottom: 15px;
                    color: #555;
                }}
                
                .error-icon {{
                    font-size: 40px;
                    color: white;
                    margin-bottom: 20px;
                    background: var(--error-color);
                    width: 80px;
                    height: 80px;
                    line-height: 80px;
                    border-radius: 50%;
                    display: inline-block;
                }}
                
                .files-list {{
                    background: var(--light-bg);
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    text-align: left;
                    max-height: 200px;
                    overflow-y: auto;
                }}
                
                .files-list code {{
                    font-family: monospace;
                    color: #666;
                }}
                
                .btn {{
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 8px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: var(--transition);
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    text-decoration: none;
                    font-weight: 500;
                    margin-top: 10px;
                }}
                
                .btn:hover {{
                    background: #3a5be0;
                    transform: translateY(-2px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <i class="fas fa-exclamation-triangle error-icon"></i>
                <h2>Error: Output File Not Found</h2>
                <p>We couldn't find the upscaled image in our system.</p>
                <p>This might be due to an error during the upscaling process or the file may have been moved or deleted.</p>
                
                <div class="files-list">
                    <p>Available files in output directory:</p>
                    <code>{', '.join(available_files) if available_files else 'No files found'}</code>
                </div>
                
                <a href="/" class="btn">
                    <i class="fas fa-redo"></i> Try Again
                </a>
            </div>
        </body>
        </html>
        '''

@app.route('/cleanup', methods=['GET', 'POST'])
def cleanup():
    """Endpoint to handle cleanup requests via beacon API"""
    cleanup_immediately()
    SESSION_FILES.clear()
    return '', 204  # Return empty response with "No Content" status

@app.route('/download/<filename>')
def download_file(filename):
    # After download, schedule file for cleanup
    session_id = request.cookies.get('session', str(time.time()))
    if session_id not in SESSION_FILES:
        SESSION_FILES[session_id] = []
    
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        SESSION_FILES[session_id].append(file_path)
    
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# Define additional static folders
@app.route('/static/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

# Ensure the images directory exists
os.makedirs('static/images', exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)
