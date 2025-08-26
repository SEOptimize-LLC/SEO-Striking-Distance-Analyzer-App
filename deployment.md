# Deployment Guide

## Streamlit Cloud Deployment

1. **Prepare Repository**:
   - Ensure app.py and requirements.txt are in root directory
   - Commit all files to GitHub repository

2. **Deploy to Streamlit Cloud**:
   - Go to share.streamlit.io
   - Connect your GitHub account
   - Select repository and branch
   - Set main file path to: app.py
   - Deploy

3. **Configuration**:
   - The app will automatically install requirements
   - No additional configuration needed

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File Structure

```
striking-distance-analyzer/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── deployment.md      # This file
```
