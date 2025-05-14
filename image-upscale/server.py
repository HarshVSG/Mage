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
    
    file = request.files['image']
    model = request.form.get('model', 'RealESRGAN_x4plus')
    resolution = request.form.get('resolution', '4')  # Default to 4x if not specified
    
    if not file:
        return 'No file uploaded.', 400

    if model not in MODELS:
        return 'Invalid model selected.', 400
        
    if resolution not in RESOLUTIONS:
        return 'Invalid resolution selected.', 400
        
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    # Track this file in our session
    session_id = request.cookies.get('session', str(time.time()))
    if session_id not in SESSION_FILES:
        SESSION_FILES[session_id] = []
    SESSION_FILES[session_id].append(input_path)
    
    name, ext = os.path.splitext(filename)
    # The Real-ESRGAN script will create the output with this name pattern by default
    output_filename = f'{name}_out{ext}'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    command = [
        "python", "Real-ESRGAN/inference_realesrgan.py",
        "-n", model,
        "-i", input_path,
        "-o", OUTPUT_FOLDER,
        "--fp32",  # Use full precision to avoid potential issues
        "-s", resolution  # Use the selected resolution scale factor
    ]
    
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
    print(f"Command output: {result.stdout}")
    print(f"Command error: {result.stderr}")
    
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
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mage - Upscaled Result</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            {cleanup_script}
            <style>
                :root {{
                    --primary-color: #4a6cf7;
                    --primary-hover: #3a5be0;
                    --secondary-color: #6c757d;
                    --success-color: #4BB543;
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
                    max-width: 1000px;
                    width: 100%;
                    background-color: white;
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                    padding: 30px;
                    margin-top: 20px;
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
                    background: linear-gradient(90deg, var(--primary-color), #8e2de2);
                }}
                
                h1, h2 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                    font-weight: 600;
                }}
                
                .success-header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                
                .success-icon {{
                    text-align: center;
                    margin-bottom: 20px;
                    background: var(--light-bg);
                    width: 80px;
                    height: 80px;
                    line-height: 80px;
                    border-radius: 50%;
                    display: inline-block;
                }}
                
                .success-icon i {{
                    font-size: 40px;
                    color: var(--success-color);
                }}
                
                .model-badge {{
                    display: inline-block;
                    background: var(--light-bg);
                    padding: 8px 15px;
                    border-radius: 20px;
                    margin-top: 10px;
                    font-size: 14px;
                    color: var(--primary-color);
                }}
                
                .image-comparison {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 30px;
                    margin-bottom: 40px;
                }}
                
                .image-box {{
                    flex: 1;
                    min-width: 300px;
                    background-color: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 20px;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                    transition: var(--transition);
                }}
                
                .image-box:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                }}
                
                .image-box h3 {{
                    text-align: center;
                    margin-bottom: 15px;
                    color: #333;
                    font-weight: 500;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                }}
                
                .image-box h3 i {{
                    color: var(--primary-color);
                }}
                
                .image-box img {{
                    max-width: 100%;
                    border-radius: 8px;
                    display: block;
                    margin: 0 auto;
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
                }}
                
                .details {{
                    margin-top: 15px;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                }}
                
                .detail-item {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                    font-size: 14px;
                    color: #666;
                }}
                
                .detail-item:last-child {{
                    margin-bottom: 0;
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
                    justify-content: center;
                    gap: 8px;
                    font-weight: 500;
                    text-decoration: none;
                }}
                
                .btn:hover {{
                    background: var(--primary-hover);
                    transform: translateY(-2px);
                }}
                
                .btn-container {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    flex-wrap: wrap;
                    margin: 30px 0;
                }}
                
                .feature-list {{
                    margin-top: 40px;
                    background-color: var(--light-bg);
                    border-radius: var(--border-radius);
                    padding: 25px;
                }}
                
                .feature-list h3 {{
                    text-align: center;
                    margin-bottom: 20px;
                    color: #333;
                    font-weight: 500;
                }}
                
                .features-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                }}
                
                .feature-item {{
                    display: flex;
                    align-items: flex-start;
                    gap: 10px;
                }}
                
                .feature-item i {{
                    color: var(--success-color);
                    font-size: 18px;
                    margin-top: 2px;
                }}
                
                .feature-content h4 {{
                    font-size: 16px;
                    margin-bottom: 5px;
                    color: #333;
                }}
                
                .feature-content p {{
                    font-size: 14px;
                    color: #666;
                }}
                
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 20px;
                    }}
                    
                    .image-comparison {{
                        flex-direction: column;
                    }}
                    
                    .btn-container {{
                        flex-direction: column;
                    }}
                    
                    .btn {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-header">
                    <div class="success-icon">
                        <i class="fas fa-check"></i>
                    </div>
                    <h1>Image Successfully Upscaled!</h1>
                    <div class="model-badge">
                        <i class="fas fa-wand-magic-sparkles"></i> Upscaled with {model_name} at {resolution_name}
                    </div>
                </div>
                
                <div class="image-comparison">
                    <div class="image-box">
                        <h3><i class="fas fa-image"></i> Original Image</h3>
                        <img src="/static/uploads/{original}" alt="Original Image">
                        <div class="details">
                            <div class="detail-item">
                                <span>File Size</span>
                                <span>{original_size:.2f} MB</span>
                            </div>
                            <div class="detail-item">
                                <span>Dimensions</span>
                                <span>{original_dimensions}</span>
                            </div>
                            <div class="detail-item">
                                <span>Format</span>
                                <span>{os.path.splitext(original)[1].upper().replace('.', '')}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="image-box">
                        <h3><i class="fas fa-wand-magic-sparkles"></i> Upscaled Image</h3>
                        <img src="/static/outputs/{filename}" alt="Upscaled Image">
                        <div class="details">
                            <div class="detail-item">
                                <span>File Size</span>
                                <span>{output_size:.2f} MB</span>
                            </div>
                            <div class="detail-item">
                                <span>Dimensions</span>
                                <span>{output_dimensions}</span>
                            </div>
                            <div class="detail-item">
                                <span>Format</span>
                                <span>{os.path.splitext(filename)[1].upper().replace('.', '')}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="btn-container">
                    <a href="/download/{filename}" class="btn">
                        <i class="fas fa-download"></i> Download Upscaled Image
                    </a>
                    <a href="/" class="btn">
                        <i class="fas fa-redo"></i> Upscale Another Image
                    </a>
                </div>
                
                <div class="feature-list">
                    <h3>About Mage AI Image Upscaler</h3>
                    <div class="features-grid">
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>Advanced AI Models</h4>
                                <p>Powered by Real-ESRGAN technology</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>4x Resolution</h4>
                                <p>Increases image size by up to 4 times</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>Detail Enhancement</h4>
                                <p>Recovers and sharpens fine details</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>Noise Reduction</h4>
                                <p>Removes compression artifacts</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>Specialized Models</h4>
                                <p>Options for photos, anime, and art</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <i class="fas fa-check-circle"></i>
                            <div class="feature-content">
                                <h4>Free to Use</h4>
                                <p>No watermarks or limits</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Powered by Real-ESRGAN AI Technology | Mage AI Image Enhancer</p>
                <p>© {datetime.datetime.now().year} Mage</p>
            </div>
        </body>
        </html>
        '''
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

if __name__ == '__main__':
    app.run(debug=True)
