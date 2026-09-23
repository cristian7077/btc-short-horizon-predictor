WEB VERSION

1. Put web_server.py and web.html in your existing btc_short_horizon_predictor-3 folder.
2. Make sure the folder has models/btc_1m.joblib, btc_5m.joblib, btc_15m.joblib.
3. Install:
   pip install fastapi uvicorn
4. Run:
   uvicorn web_server:app --reload
5. Open:
   http://127.0.0.1:8000

The page refreshes its market snapshot every second. It uses the existing Python model files and feature code.
