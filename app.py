from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import threading
import time
import heapq
import io
from ultralytics import YOLO
import logging
import requests
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("traffic_monitor.log")
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =====================================================================
# CONSTANTS AND CONFIGURATION
# =====================================================================

# Load YOLO model
try:
    model = YOLO('yolov8l.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

# Traffic status thresholds
STATUS_THRESHOLDS = {
    'lancar': 5,   # Smooth traffic (≤ 5 vehicles)
    'padat': 15,   # Dense traffic (≤ 15 vehicles)
    'macet': 20    # Congested traffic (> 15 vehicles)
}

# Vehicle classes to detect
ALLOWED_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

# Vehicle color mapping for visualization
VEHICLE_COLOR_MAP = {
    'car': (0, 255, 0),       # Green
    'truck': (0, 0, 255),     # Red
    'bus': (255, 0, 0),       # Blue
    'motorcycle': (255, 255, 0) # Yellow
}

# CCTV video paths
VIDEO_PATHS = [
    "https://livepantau.semarangkota.go.id/hls/414/701/2024/8795266c-ebc3-4a95-8827-b82360403f0a_701.m3u8",  # 0. kalibanteng
    "https://livepantau.semarangkota.go.id/hls/414/1101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_110139.ts",  # 1. kaligarang
    "https://livepantau.semarangkota.go.id/hls/414/4801/2024/8795266c-ebc3-4a95-8827-b82360403f0a_4801.m3u8",  # 2. madukuro
    "https://livepantau.semarangkota.go.id/hls/414/901/2024/8795266c-ebc3-4a95-8827-b82360403f0a_901.m3u8",  # 3. kariadi
    "https://livepantau.semarangkota.go.id/hls/414/101/2024/8795266c-ebc3-4a95-8827-b82360403f0a_10119.ts",  # 4. tugu muda
    "https://livepantau.semarangkota.go.id/hls/414/5201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_520120.ts",  # 5. indraprasta
    "https://livepantau.semarangkota.go.id/hls/414/5301/2024/8795266c-ebc3-4a95-8827-b82360403f0a_5301.m3u8",  # 6. bergota
    "https://livepantau.semarangkota.go.id/hls/414/201/2024/8795266c-ebc3-4a95-8827-b82360403f0a_201.m3u8",  # 7. simp kyai saleh
]

# Node positions for graph visualization
POSITIONS = {
    0: (0, 0), 1: (1, 2), 2: (2, 0), 3: (3, 2),
    4: (4, 0), 5: (5, 2), 6: (6, 0), 7: (7, 2)
}

# Node labels (location names)
NODE_LABELS = {
    0: "Kalibanteng", 1: "Kaligarang", 2: "Madukuro", 3: "Kariadi",
    4: "Tugu Muda", 5: "Indraprasta", 6: "Bergota", 7: "Simpang Kyai Saleh"
}

# Road connections (graph edges)
EDGES = [
    (0, 1),  # 0 terhubung ke 1
    (0, 2),  # 0 terhubung ke 2
    (2, 5),  # 2 terhubung ke 5
    (2, 1),  # 2 terhubung ke 1
    (2, 4),  # 2 terhubung ke 4
    (5, 4),  # 5 terhubung ke 4
    (1, 3),  # 1 terhubung ke 3
    (4, 3),  # 4 terhubung ke 3
    (4, 7),  # 4 terhubung ke 7
    (7, 6),  # 7 terhubung ke 6
    (3, 6)   # 3 terhubung ke 6
]

# =====================================================================
# VIDEO STREAM HANDLER CLASS
# =====================================================================

class VideoStreamHandler:
    """
    A class to handle video stream processing with robust error handling and frame buffering.
    """
    def __init__(self, video_path, node_id, buffer_size=5, max_retries=10, retry_delay=3, 
                 reconnect_threshold=10, timeout=10):
        """
        Initialize the video stream handler.
        
        Args:
            video_path (str): URL or path to the video stream
            node_id (int): ID of the node/location
            buffer_size (int): Number of frames to buffer
            max_retries (int): Maximum number of connection retry attempts
            retry_delay (int): Delay between retry attempts in seconds
            reconnect_threshold (int): Number of consecutive errors before reconnecting
            timeout (int): Connection timeout in seconds
        """
        self.video_path = video_path
        self.node_id = node_id
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reconnect_threshold = reconnect_threshold
        self.timeout = timeout
        
        self.frame_buffer = deque(maxlen=buffer_size)
        self.is_running = True
        self.error_count = 0
        self.total_errors = 0
        self.cap = None
        self.last_frame = self._create_placeholder_frame(f"Starting stream for {NODE_LABELS[node_id]}...")
        
        # Stats tracking
        self.vehicle_count = 0
        self.traffic_status = "lancar"
        
    def _create_placeholder_frame(self, message):
        """Create a placeholder frame with a message when stream is unavailable."""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background gradient for better visibility
        for y in range(height):
            color_value = int(180 * (y / height)) + 50
            frame[y, :] = [color_value, color_value, color_value]
            
        # Add text with message
        cv2.putText(frame, NODE_LABELS[self.node_id], (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 160, 255), 2)
        
        cv2.putText(frame, message, (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        cv2.putText(frame, f"Retrying... ({self.total_errors})", (20, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                    
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 230, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
        return frame
        
    def _init_capture(self):
        """Initialize or reinitialize the video capture with proper error handling."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
                
        try:
            # Check if file or URL exists before attempting capture
            if self.video_path.startswith(('http://', 'https://')):
                try:
                    response = requests.head(self.video_path, timeout=self.timeout)
                    if response.status_code >= 400:
                        logger.warning(f"Stream URL returns status code {response.status_code} for node {self.node_id}")
                except Exception as e:
                    logger.warning(f"Cannot connect to stream URL for node {self.node_id}: {str(e)}")
            
            # Initialize capture with a timeout
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Check if capture is successfully opened
            if not self.cap.isOpened():
                logger.error(f"Failed to open video stream for node {self.node_id}")
                self.last_frame = self._create_placeholder_frame("Stream unavailable")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture for node {self.node_id}: {str(e)}")
            self.last_frame = self._create_placeholder_frame(f"Error: {str(e)[:30]}...")
            return False
            
    def _process_frame(self, frame):
        """Process a frame with YOLO detection and add annotations."""
        try:
            # YOLO Inference
            results = model.predict(frame, stream=True)
            vehicle_count = 0  # Reset vehicle count
            
            # Process detection results
            processed_frame = frame.copy()
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])  # Class ID
                    label = model.names[cls]
                    
                    # Skip non-vehicle classes
                    if label not in ALLOWED_VEHICLE_CLASSES:
                        continue
                        
                    # Extract bounding box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0]
                    vehicle_count += 1
                    
                    # Get color for this vehicle type
                    color = VEHICLE_COLOR_MAP.get(label, (255, 255, 255))
                    
                    # Draw bounding box and label
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(processed_frame, text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Determine traffic status based on vehicle count
            if vehicle_count <= STATUS_THRESHOLDS['lancar']:
                traffic_status = "Lancar"
                status_color = (0, 255, 0)  # Green for smooth traffic
            elif vehicle_count <= STATUS_THRESHOLDS['padat']:
                traffic_status = "Padat"
                status_color = (0, 165, 255)  # Orange for dense traffic
            else:
                traffic_status = "Macet"
                status_color = (0, 0, 255)  # Red for congested traffic
                
            # Add annotation overlay
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
            
            # Add status and vehicle count text
            cv2.putText(processed_frame, f"{NODE_LABELS[self.node_id]}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Status: {traffic_status}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(processed_frame, f"Kendaraan: {vehicle_count}", (220, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                       
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, 
                       (processed_frame.shape[1] - 230, processed_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Update vehicle count and traffic status
            self.vehicle_count = vehicle_count
            self.traffic_status = traffic_status.lower()
            
            return processed_frame, vehicle_count
            
        except Exception as e:
            logger.error(f"Error processing frame for node {self.node_id}: {str(e)}")
            return frame, 0
    
    def start(self):
        """Start the video processing thread."""
        threading.Thread(target=self._process_video_stream, daemon=True).start()
        
    def _process_video_stream(self):
        """Main processing loop for the video stream with error handling and retry logic."""
        retry_count = 0
        
        while self.is_running and retry_count < self.max_retries:
            # Try to initialize capture
            if not self._init_capture():
                retry_count += 1
                self.total_errors += 1
                logger.warning(f"Failed to initialize capture for node {self.node_id}, "
                               f"retry {retry_count}/{self.max_retries}")
                time.sleep(self.retry_delay)
                continue
                
            # Reset retry count once we have a successful connection
            retry_count = 0
            self.error_count = 0
            
            # Main processing loop for frames
            while self.is_running:
                try:
                    # Read frame with timeout protection
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        self.error_count += 1
                        self.total_errors += 1
                        logger.warning(f"Failed to read frame from node {self.node_id}, "
                                      f"error count: {self.error_count}")
                        
                        # Use the last valid frame from buffer if available
                        if self.frame_buffer:
                            self.last_frame = self.frame_buffer[-1]
                        else:
                            self.last_frame = self._create_placeholder_frame("Connection interrupted")
                            
                        # Check if we need to reconnect
                        if self.error_count >= self.reconnect_threshold:
                            logger.info(f"Reconnecting to stream for node {self.node_id} after "
                                       f"{self.error_count} consecutive errors")
                            break
                            
                        time.sleep(1)
                        continue
                        
                    # Successful frame read, reset error count
                    self.error_count = 0
                    
                    # Process the frame
                    processed_frame, vehicle_count = self._process_frame(frame)
                    
                    # Update the latest frame
                    self.last_frame = processed_frame
                    
                    # Add to buffer
                    self.frame_buffer.append(processed_frame)
                    
                    # Update global traffic graph
                    update_edge_weights(self.node_id, vehicle_count)
                    
                    # Throttle processing to reduce CPU usage
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.error_count += 1
                    self.total_errors += 1
                    logger.error(f"Error in processing stream for node {self.node_id}: {str(e)}")
                    
                    # Create placeholder on error
                    self.last_frame = self._create_placeholder_frame(f"Processing error: {str(e)[:30]}...")
                    
                    # Break out of inner loop to reconnect
                    if self.error_count >= self.reconnect_threshold:
                        break
                        
                    time.sleep(self.retry_delay)
            
            # Clean up capture before retrying
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass
                
            time.sleep(self.retry_delay)
        
        # All retries exhausted
        if retry_count >= self.max_retries:
            logger.error(f"Maximum retry count reached for node {self.node_id}, giving up")
            self.last_frame = self._create_placeholder_frame("Stream unavailable after max retries")
        
    def get_latest_frame(self):
        """Get the latest processed frame."""
        return self.last_frame
        
    def get_vehicle_count(self):
        """Get the current vehicle count."""
        return self.vehicle_count
        
    def get_traffic_status(self):
        """Get the current traffic status."""
        return self.traffic_status
        
    def stop(self):
        """Stop the processing thread."""
        self.is_running = False
        if self.cap:
            self.cap.release()

# =====================================================================
# GRAPH AND ROUTE CALCULATION
# =====================================================================

# Initialize graph
graph = nx.Graph()
graph.add_nodes_from(POSITIONS.keys())
graph.add_edges_from(EDGES, weight=1)

# Traffic status tracking for edges
traffic_status = {edge: "lancar" for edge in graph.edges()}

# Function to update edge weights based on traffic conditions
def update_edge_weights(node_id, vehicle_count):
    """
    Update the edge weights in the graph based on detected vehicle count.
    
    Args:
        node_id (int): The node ID where vehicle count was detected
        vehicle_count (int): Number of vehicles detected
    """
    # Determine travel time based on vehicle count
    travel_time = (
        1 if vehicle_count <= STATUS_THRESHOLDS['lancar'] else
        3 if vehicle_count <= STATUS_THRESHOLDS['padat'] else
        5
    )
    
    # Update weights for all edges connected to this node
    for neighbor in graph.neighbors(node_id):
        graph[node_id][neighbor]['weight'] = travel_time
        
        # Update traffic status for the edge
        edge = (node_id, neighbor) if (node_id, neighbor) in traffic_status else (neighbor, node_id)
        traffic_status[edge] = (
            "lancar" if vehicle_count <= STATUS_THRESHOLDS['lancar'] else
            "padat" if vehicle_count <= STATUS_THRESHOLDS['padat'] else
            "macet"
        )

# Dijkstra algorithm implementation
def dijkstra(graph, start):
    """
    Implementation of Dijkstra's algorithm for finding shortest paths.
    
    Args:
        graph (networkx.Graph): The road network graph
        start (int): Starting node ID
        
    Returns:
        tuple: (dist, prev) where dist is a dict of distances and prev tracks the path
    """
    dist = {node: float('inf') for node in graph.nodes()}
    dist[start] = 0
    pq = [(0, start)]
    prev = {node: None for node in graph.nodes()}

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_dist > dist[current_node]:
            continue
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_dist + weight
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return dist, prev

# Function to recommend best route
def recommend_route(start, end):
    """
    Calculate the recommended route between two nodes.
    
    Args:
        start (int): Starting node ID
        end (int): Destination node ID
        
    Returns:
        tuple: (path, distance) where path is a list of node IDs and distance is the total cost
    """
    dist, prev = dijkstra(graph, start)
    path = []
    current_node = end
    
    # Reconstruct the path
    while current_node is not None:
        path.append(current_node)
        current_node = prev[current_node]
    path.reverse()
    
    return path, dist[end]

# Function to visualize the graph
def visualize_graph(start=0, end=7):
    """
    Create a visualization of the road network graph with the best route highlighted.
    
    Args:
        start (int): Starting node ID
        end (int): Destination node ID
        
    Returns:
        tuple: (graph_image, best_route) where graph_image is a base64 encoded image
    """
    plt.figure(figsize=(12, 7))
    
    # Determine edge colors based on traffic conditions
    edge_colors = []
    edge_widths = []
    for edge in graph.edges(data=True):
        weight = edge[2]['weight']
        if weight == 1:  # Lancar
            edge_colors.append("#3CB371")  # Green
            edge_widths.append(3)
        elif weight == 3:  # Padat
            edge_colors.append("#FFD700")  # Yellow
            edge_widths.append(4)
        else:  # Macet
            edge_colors.append("#FF6347")  # Red
            edge_widths.append(5)

    # Highlight the best route in blue
    best_route, _ = recommend_route(start, end)
    best_edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
    
    # Draw graph background
    plt.gca().set_facecolor('#F5F5F5')  # Light gray background
    
    # Draw base nodes and edges
    nx.draw(graph, pos=POSITIONS, with_labels=False, node_size=800, 
            node_color="#B0C4DE", edgecolors='black', linewidths=1.5)
            
    # Draw edges with color coding
    nx.draw_networkx_edges(graph, pos=POSITIONS, edge_color=edge_colors, width=edge_widths)
    
    # Draw best route edges
    if best_edges:
        nx.draw_networkx_edges(graph, pos=POSITIONS, edgelist=best_edges, 
                              edge_color='#4169E1', width=6, alpha=0.8)
                              
    # Draw node labels
    label_positions = {node: (pos[0], pos[1] - 0.1) for node, pos in POSITIONS.items()}
    nx.draw_networkx_labels(graph, pos=label_positions, labels=NODE_LABELS, 
                           font_size=11, font_weight='bold', font_color='black')
    
    # Add traffic status legend
    plt.plot([], [], color="#3CB371", linewidth=3, label='Lancar')
    plt.plot([], [], color="#FFD700", linewidth=3, label='Padat')
    plt.plot([], [], color="#FF6347", linewidth=3, label='Macet')
    plt.plot([], [], color="#4169E1", linewidth=4, label='Rute Terbaik')
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')
    
    # Mark start and end nodes
    nx.draw_networkx_nodes(graph, pos=POSITIONS, nodelist=[start], 
                          node_color='#32CD32', node_size=900, edgecolors='black')
    nx.draw_networkx_nodes(graph, pos=POSITIONS, nodelist=[end], 
                          node_color='#FF6347', node_size=900, edgecolors='black')
    
    # Add title
    plt.title(f"Rute dari {NODE_LABELS[start]} ke {NODE_LABELS[end]}", fontsize=14, fontweight='bold')
    
    # Remove axis
    plt.axis('off')
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Encode the buffer to base64
    graph_image = base64.b64encode(buf.read()).decode('utf-8')
    return graph_image, best_route

# =====================================================================
# FLASK ROUTES
# =====================================================================

# Initialize video stream handlers
stream_handlers = {}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', node_labels=NODE_LABELS)

@app.route('/get_frames')
def get_frames():
    """Return the latest frames from all CCTV streams."""
    encoded_frames = {}
    
    for node_id, handler in stream_handlers.items():
        frame = handler.get_latest_frame()
        
        encoded_frames[node_id] = {
            'image': encode_frame(frame),
            'vehicle_count': handler.get_vehicle_count(),
            'status': handler.get_traffic_status().capitalize()
        }
        
    return jsonify(encoded_frames)

@app.route('/get_route', methods=['POST'])
def get_route():
    """Calculate and return the best route between two points."""
    try:
        data = request.get_json()
        start_node = int(data['start'])
        end_node = int(data['end'])
        
        # Generate graph visualization
        graph_image, route = visualize_graph(start_node, end_node)
        
        # Build route details
        route_text = []
        for i in range(len(route) - 1):
            current = route[i]
            next_node = route[i + 1]
            
            # Get traffic status for this segment
            edge = (current, next_node) if (current, next_node) in traffic_status else (next_node, current)
            status = traffic_status.get(edge, "lancar")
            
            route_text.append({
                'from': NODE_LABELS[current],
                'to': NODE_LABELS[next_node],
                'status': status
            })
        
        # Return response with graph image and route details
        return jsonify({
            'graph_image': graph_image,
            'route': route_text,
            'route_nodes': [NODE_LABELS[node] for node in route]
        })
        
    except Exception as e:
        logger.error(f"Error calculating route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Helper function to encode a frame to base64
def encode_frame(frame):
    """
    Encode a frame as base64 for transmission to the client.
    
    Args:
        frame (numpy.ndarray): The frame to encode
        
    Returns:
        str: Base64 encoded JPEG image
    """
    if frame is None:
        # Return a black frame if no frame is available
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", black_frame)
    else:
        # Resize frame for better performance
        resized_frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return base64.b64encode(buffer).decode('utf-8')

# =====================================================================
# APPLICATION INITIALIZATION
# =====================================================================

def initialize_system():
    """Initialize the entire monitoring system."""
    logger.info("Initializing traffic monitoring system...")
    
    # Initialize stream handlers for each CCTV location
    for node_id, video_path in enumerate(VIDEO_PATHS):
        logger.info(f"Initializing stream for node {node_id}: {NODE_LABELS[node_id]}")
        handler = VideoStreamHandler(video_path, node_id)
        stream_handlers[node_id] = handler
        handler.start()
        
    logger.info("All video streams initialized")

if __name__ == '__main__':
    # Initialize the system
    initialize_system()
    
    # Start the Flask application
    logger.info("Starting Flask web server...")
    app.run(debug=True, threaded=True, host='0.0.0.0')
    
    # Cleanup on exit
    for handler in stream_handlers.values():
        handler.stop()