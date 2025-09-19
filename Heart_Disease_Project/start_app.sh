#!/bin/bash
# Heart Disease Prediction App Startup Script

echo "🚀 Starting Heart Disease Prediction System..."

# Check if we're in the right directory
if [ ! -f "ui/app.py" ]; then
    echo "❌ Error: Please run this script from the Heart_Disease_Project directory"
    exit 1
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit is not installed. Please run: pip install streamlit"
    exit 1
fi

# Check if Ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Error: Ngrok is not installed. Please install Ngrok first."
    exit 1
fi

echo "✅ All dependencies found"

# Start Streamlit in background
echo "🔄 Starting Streamlit application..."
streamlit run ui/app.py --server.headless true --server.port 8501 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
sleep 5

# Check if Streamlit is running
if ! curl -s http://localhost:8501 > /dev/null; then
    echo "❌ Error: Streamlit failed to start"
    kill $STREAMLIT_PID 2>/dev/null
    exit 1
fi

echo "✅ Streamlit is running on http://localhost:8501"

# Start Ngrok
echo "🔄 Starting Ngrok tunnel..."
ngrok http 8501 &
NGROK_PID=$!

# Wait for Ngrok to start
sleep 3

echo "✅ Ngrok tunnel started"
echo ""
echo "🌐 Your Heart Disease Prediction App is now live!"
echo "🏠 Local URL: http://localhost:8501"
echo "🌍 Public URL: Check the Ngrok output above for the public URL"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait
