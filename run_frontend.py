import http.server
import socketserver
import os
import webbrowser
import threading
import time

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def open_browser():
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:3000/ProctorVision.html')

if __name__ == "__main__":
    PORT = 3000
    os.chdir("C:\\Users\\shrut\\Desktop\\ProctorVision")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f" ProctorVision Frontend Server running on http://localhost:{PORT}/")
        print(f" Visit: http://localhost:{PORT}/ProctorVision.html")
        print(f" Backend API running on http://localhost:8002")
        print(f" Opening browser automatically...")
        httpd.serve_forever()