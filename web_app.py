from flask import Flask, jsonify, send_from_directory, request
import os
import database

app = Flask(__name__, static_folder='static')

# Configuration
SNAPSHOTS_DIR = os.path.abspath('snapshots')
WEB_PORT = int(os.getenv('WEB_PORT', 5050))

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/observations')
def get_observations():
    """Get all observations with pagination"""
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        observations = database.get_all_observations(limit=limit, offset=offset)
        return jsonify({
            'success': True,
            'data': observations,
            'count': len(observations)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/observations/<int:obs_id>')
def get_observation(obs_id):
    """Get a single observation by ID"""
    try:
        observation = database.get_observation_by_id(obs_id)
        if observation:
            return jsonify({
                'success': True,
                'data': observation
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Observation not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_statistics():
    """Get summary statistics"""
    try:
        stats = database.get_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/hourly-averages')
def get_hourly_stats():
    """Get hourly average data for charts"""
    try:
        hourly_data = database.get_hourly_averages()
        return jsonify({
            'success': True,
            'data': hourly_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    """Serve snapshot images"""
    try:
        return send_from_directory(SNAPSHOTS_DIR, filename)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'traffic-watcher-web'
    })

if __name__ == '__main__':
    # Initialize database
    database.init_db()
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    print(f"Starting web interface on port {WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, debug=True)
