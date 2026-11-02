"""SENTINEL: Surveillance Forensics & Incident Retrieval Web Console.

Launch the web console:
    python app.py
    (or: python server.py)

Console runs at: http://localhost:8000
"""

import os
import sys
import webbrowser
import uvicorn
from server import app

if __name__ == "__main__":
    port = 8000
    url = f"http://localhost:{port}"
    print("\n" + "=" * 65)
    print("SENTINEL Surveillance Forensics Console")
    print(f"Opening dashboard at: {url}")
    print("=" * 65 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
elif len(sys.argv) > 0 and "streamlit" in sys.argv[0].lower():
    # Only if executed via 'streamlit run app.py'
    try:
        import streamlit as st
        st.set_page_config(page_title="SENTINEL Console", layout="wide")
        st.markdown(
            """
            <div style="background:#151617;color:#D8DADC;padding:24px;border:1px solid #26282A;border-radius:2px;font-family:'IBM Plex Sans',system-ui,sans-serif;text-align:center;">
                <h2 style="font-size:18px;font-weight:600;margin-bottom:8px;">SENTINEL Forensic Examination Console</h2>
                <p style="color:#8C9094;font-size:13px;">This application has been upgraded to a standalone forensic workstation running on FastAPI.</p>
                <p style="font-size:14px;margin:20px 0;color:#D9A441;">Stop Streamlit (Ctrl+C in terminal) and run:</p>
                <div style="background:#0A0B0C;border:1px solid #3A3D40;padding:8px 16px;border-radius:2px;font-family:'IBM Plex Mono',monospace;font-size:14px;display:inline-block;color:#D8DADC;">
                    python app.py
                </div>
                <p style="color:#8C9094;margin-top:16px;font-size:12px;">Console URL: <a href="http://localhost:8000" target="_blank" style="color:#6E8CA3;text-decoration:none;">http://localhost:8000</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass
