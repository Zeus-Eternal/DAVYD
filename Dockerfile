# ----------------------
# DAVYD - Dynamic AI Virtual Yielding Dataset
# Dockerfile for Streamlit-based Application
# Developer: agustealo | agustealo.com
# ----------------------

# Use an official Python 3.9 slim image as the base
FROM python:3.9-slim

# Metadata for branding
LABEL maintainer="agustealo <agustealo@gmail.com>"
LABEL version="1.0.0"
LABEL description="DAVYD - Intelligent Dataset Generator Powered by AI"

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /davyd

# Branding: Add logo/banner (optional)
COPY static/banner.txt /davyd/

# Copy project dependencies
COPY requirements.txt /davyd/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Branding: Display the banner
RUN echo "Welcome to DAVYD by agustealo!" && \
    cat /davyd/banner.txt

# Copy the entire project
COPY . /davyd/

# Expose the default Streamlit port
EXPOSE 8501

# Branding: Log container start
RUN echo "DAVYD is ready to launch. Visit http://localhost:8501"

# Start the Streamlit application
CMD ["streamlit", "run", "src/ui.py"]

# ----------------------
# Usage Notes:
# 1. Build: docker build -t davyd:1.0.0 .
# 2. Run: docker run -p 8501:8501 davyd:1.0.0
# 3. Visit: http://localhost:8501
# ----------------------
