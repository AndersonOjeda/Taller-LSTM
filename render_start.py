"""Arranque Streamlit con el puerto asignado por Render."""
import os
import sys

os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', 'app.py',
                        '--server.address=0.0.0.0',
                        '--server.port=' + os.environ.get('PORT', '8501'),
                        '--server.headless=true',
                        '--browser.gatherUsageStats=false'])
