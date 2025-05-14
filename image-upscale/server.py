from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import subprocess
import time
import glob
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files():
    """Delete all files in upload and output folders to free up space"""
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        files = glob.glob(os.path.join(folder, '*'))
        for f in files:
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")

@app.route('/')
def index():
    # Clean up old files when returning to the homepage
    cleanup_old_files()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Clean up old files first
    cleanup_old_files()
    
    file = request.files['image']
    if not file:
        return 'No file uploaded.', 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    name, ext = os.path.splitext(filename)
    # The Real-ESRGAN script will create the output with this name pattern by default
    output_filename = f'{name}_out{ext}'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    command = [
        "python", "Real-ESRGAN/inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_path,
        "-o", OUTPUT_FOLDER,
        "--fp32"  # Use full precision to avoid potential issues
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
        # Look for any file that might have been created with a similar name
        possible_files = [f for f in os.listdir(OUTPUT_FOLDER) if name in f]
        if possible_files:
            output_filename = possible_files[0]
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            print(f"Found possible output file: {output_path}")

    return redirect(url_for('show_result', filename=output_filename, original=filename))

@app.route('/result/<filename>')
def show_result(filename):
    # Get original filename
    original = request.args.get('original', '')
    
    # Check if files exist
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    original_path = os.path.join(UPLOAD_FOLDER, original)
    
    if os.path.exists(output_path):
        # Get file sizes
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # in MB
        original_size = os.path.getsize(original_path) / (1024 * 1024) if os.path.exists(original_path) else 0
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upscaled Result</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Roboto', sans-serif;
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
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    padding: 30px;
                    margin-top: 20px;
                }}
                h1, h2 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .success-icon {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .success-icon i {{
                    font-size: 60px;
                    color: #4BB543;
                }}
                .image-comparison {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .image-box {{
                    flex: 1;
                    min-width: 300px;
                    background-color: #f8f9ff;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                }}
                .image-box h3 {{
                    text-align: center;
                    margin-bottom: 10px;
                    color: #333;
                }}
                .image-box img {{
                    max-width: 100%;
                    border-radius: 5px;
                    display: block;
                    margin: 0 auto;
                }}
                .details {{
                    display: flex;
                    justify-content: space-between;
                    margin-top: 10px;
                    font-size: 14px;
                    color: #666;
                    padding: 0 10px;
                }}
                .btn {{
                    background: #4a6cf7;
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 20px;
                    transition: all 0.3s;
                    display: inline-block;
                    text-decoration: none;
                    text-align: center;
                }}
                .btn:hover {{
                    background: #3a5be0;
                    transform: translateY(-2px);
                }}
                .btn-container {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    flex-wrap: wrap;
                }}
                .feature-list {{
                    margin-top: 40px;
                    background-color: #f8f9ff;
                    border-radius: 10px;
                    padding: 20px;
                }}
                .feature-list h3 {{
                    text-align: center;
                    margin-bottom: 15px;
                    color: #333;
                }}
                .feature-list ul {{
                    list-style-type: none;
                    padding-left: 20px;
                }}
                .feature-list li {{
                    margin-bottom: 10px;
                    position: relative;
                    padding-left: 30px;
                }}
                .feature-list li i {{
                    color: #4BB543;
                    position: absolute;
                    left: 0;
                    top: 2px;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-icon">
                    <i class="fas fa-check-circle"></i>
                </div>
                <h1>Image Successfully Upscaled!</h1>
                
                <div class="image-comparison">
                    <div class="image-box">
                        <h3>Original Image</h3>
                        <img src="/static/uploads/{original}" alt="Original Image">
                        <div class="details">
                            <span>Original Size</span>
                            <span>{original_size:.2f} MB</span>
                        </div>
                    </div>
                    
                    <div class="image-box">
                        <h3>Upscaled Image</h3>
                        <img src="/static/outputs/{filename}" alt="Upscaled Image">
                        <div class="details">
                            <span>Enhanced Size</span>
                            <span>{output_size:.2f} MB</span>
                        </div>
                    </div>
                </div>
                
                <div class="btn-container">
                    <a href="/download/{filename}" class="btn"><i class="fas fa-download"></i> Download Upscaled Image</a>
                    <a href="/" class="btn"><i class="fas fa-redo"></i> Upscale Another Image</a>
                </div>
                
                <div class="feature-list">
                    <h3>About AI Image Upscaler</h3>
                    <ul>
                        <li><i class="fas fa-check"></i> Uses advanced Real-ESRGAN AI technology</li>
                        <li><i class="fas fa-check"></i> Increases resolution while preserving details</li>
                        <li><i class="fas fa-check"></i> Removes noise and compression artifacts</li>
                        <li><i class="fas fa-check"></i> Enhances clarity and sharpness</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Powered by Real-ESRGAN AI Technology</p>
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
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Roboto', sans-serif;
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
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    padding: 30px;
                    margin-top: 20px;
                    text-align: center;
                }}
                h2 {{
                    color: #e74c3c;
                    margin-bottom: 20px;
                }}
                p {{
                    margin-bottom: 15px;
                    color: #555;
                }}
                .error-icon {{
                    font-size: 60px;
                    color: #e74c3c;
                    margin-bottom: 20px;
                }}
                .btn {{
                    background: #4a6cf7;
                    color: white;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    margin-top: 20px;
                    transition: all 0.3s;
                    display: inline-block;
                    text-decoration: none;
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
                <p>Could not find {filename} in the output folder.</p>
                <p>This might be due to an error during the upscaling process.</p>
                <p>Available files: {', '.join(available_files) if available_files else 'None'}</p>
                <a href="/" class="btn"><i class="fas fa-redo"></i> Try Again</a>
            </div>
        </body>
        </html>
        '''

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
