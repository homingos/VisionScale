#!/usr/bin/env python3
"""
Web-based Point Selector for 3D Distance Measurement
====================================================

This script provides a web interface for selecting multiple points on images
in headless environments. It creates a simple Flask server that displays
the image and allows clicking to select multiple points.

Usage:
    python web_point_selector_multi.py --image path/to/image.jpg --port 8080
"""

import argparse
import os
import json
import base64
from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Global variables to store image and points
current_image_path = None
current_image_base64 = None
selected_points = []
image_width = 0
image_height = 0

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>3D Distance Multi-Point Selector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
            position: relative;
            display: inline-block;
        }
        #imageCanvas {
            border: 2px solid #ddd;
            border-radius: 5px;
            cursor: crosshair;
            max-width: 100%;
            height: auto;
        }
        .controls {
            margin: 20px 0;
            text-align: center;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .info {
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .points-info {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .point {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: #ff4444;
            color: white;
            text-align: center;
            line-height: 20px;
            font-size: 12px;
            font-weight: bold;
            margin: 5px;
        }
        .coordinates {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
            font-size: 0.6em;
        }
        .point-counter {
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 3D Distance Multi-Point Selector</h1>
        
        <div class="info">
            <h3>Instructions:</h3>
            <ul>
                <li>Click on the image to select multiple points</li>
                <li>Points will be numbered sequentially (1, 2, 3, ...)</li>
                <li>Use "Clear Points" to start over</li>
                <li>Use "Get Coordinates" to retrieve all selected points</li>
                <li>You can select as many points as needed</li>
            </ul>
        </div>

        <div class="image-container">
            <canvas id="imageCanvas" width="{{ width }}" height="{{ height }}"></canvas>
        </div>

        <div class="controls">
            <button onclick="clearPoints()">Clear Points</button>
            <button onclick="getCoordinates()" id="getCoordsBtn" disabled>Get Coordinates</button>
            <button onclick="window.close()">Close Window</button>
        </div>

        <div class="points-info">
            <h3>Selected Points:</h3>
            <div class="point-counter" id="pointCounter">0 points selected</div>
            <div id="pointsDisplay">
                <p>No points selected yet. Click on the image to select points.</p>
            </div>
        </div>

        <div class="coordinates" id="coordinatesOutput" style="display: none;">
            <h3>Coordinates (copy these for your script):</h3>
            <pre id="coordsText"></pre>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('imageCanvas');
        const ctx = canvas.getContext('2d');
        const image = new Image();
        let points = [];
        
        image.onload = function() {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
        };
        
        image.src = 'data:image/jpeg;base64,{{ image_data }}';
        
        canvas.addEventListener('click', function(event) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Scale coordinates to actual image size
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const actualX = Math.round(x * scaleX);
            const actualY = Math.round(y * scaleY);
            
            points.push([actualX, actualY]);
            drawPoint(actualX, actualY, points.length);
            updatePointsDisplay();
            
            // Enable get coordinates button when at least one point is selected
            if (points.length >= 1) {
                document.getElementById('getCoordsBtn').disabled = false;
            }
            
            // Draw lines between consecutive points
            if (points.length >= 2) {
                drawLines();
            }
        });
        
        function drawPoint(x, y, number) {
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#ff4444';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw number
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(number, x, y);
        }
        
        function drawLines() {
            if (points.length >= 2) {
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i][0], points[i][1]);
                }
                ctx.strokeStyle = '#0066cc';
                ctx.lineWidth = 3;
                ctx.stroke();
            }
        }
        
        function clearPoints() {
            points = [];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0);
            updatePointsDisplay();
            document.getElementById('getCoordsBtn').disabled = true;
            document.getElementById('coordinatesOutput').style.display = 'none';
        }
        
        function updatePointsDisplay() {
            const display = document.getElementById('pointsDisplay');
            const counter = document.getElementById('pointCounter');
            
            counter.textContent = `${points.length} point${points.length !== 1 ? 's' : ''} selected`;
            
            if (points.length === 0) {
                display.innerHTML = '<p>No points selected yet. Click on the image to select points.</p>';
            } else {
                let html = '<div>';
                for (let i = 0; i < points.length; i++) {
                    html += `<span class="point">${i + 1}</span>`;
                    html += `<span>Point ${i + 1}: (${points[i][0]}, ${points[i][1]})</span><br>`;
                }
                html += '</div>';
                display.innerHTML = html;
            }
        }
        
        function getCoordinates() {
            if (points.length >= 1) {
                const coords = {
                    points: points,
                    command: generateCommandLine(points)
                };
                
                let coordsText = '';
                for (let i = 0; i < points.length; i++) {
                    coordsText += `Point ${i + 1}: (${points[i][0]}, ${points[i][1]})\n`;
                }
                coordsText += `\nCommand line arguments:\n${coords.command}`;
                
                document.getElementById('coordsText').textContent = coordsText;
                document.getElementById('coordinatesOutput').style.display = 'block';
                
                // Send coordinates to server
                fetch('/get_coordinates', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(coords)
                });
            }
        }
        
        function generateCommandLine(points) {
            let command = '';
            for (let i = 0; i < points.length; i++) {
                command += `--point${i + 1} ${points[i][0]} ${points[i][1]} `;
            }
            return command.trim();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    global current_image_base64, image_width, image_height
    return render_template_string(HTML_TEMPLATE, 
                                image_data=current_image_base64,
                                width=image_width,
                                height=image_height)

@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    global selected_points
    data = request.json
    selected_points = data['points']
    return jsonify({'status': 'success', 'points': selected_points})

@app.route('/api/points')
def api_points():
    global selected_points
    return jsonify({'points': selected_points})

def encode_image_to_base64(image_path):
    """Convert image to base64 string for web display."""
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def main():
    parser = argparse.ArgumentParser(description="Web-based multi-point selector for 3D distance measurement")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return
    
    global current_image_path, current_image_base64, image_width, image_height
    
    current_image_path = args.image
    current_image_base64 = encode_image_to_base64(args.image)
    
    # Get image dimensions
    img = cv2.imread(args.image)
    if img is not None:
        image_height, image_width = img.shape[:2]
    else:
        print(f"❌ Could not read image: {args.image}")
        return
    
    print(f"🌐 Starting web server on http://{args.host}:{args.port}")
    print(f"📸 Image: {args.image} ({image_width}x{image_height})")
    print(f"🎯 Instructions:")
    print(f"   1. Open your browser and go to http://localhost:{args.port}")
    print(f"   2. Click on the image to select multiple points")
    print(f"   3. Use 'Get Coordinates' to get the command line arguments")
    print(f"   4. Copy the coordinates and use them with measure_3d_distance.py")
    print(f"   5. Press Ctrl+C to stop the server")
    
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        if selected_points:
            print(f"📊 Final selected points:")
            for i, point in enumerate(selected_points):
                print(f"   Point {i+1}: ({point[0]}, {point[1]})")
            
            # Generate command line arguments
            command = ""
            for i, point in enumerate(selected_points):
                command += f"--point{i+1} {point[0]} {point[1]} "
            print(f"   Command: {command.strip()}")

if __name__ == "__main__":
    main()
