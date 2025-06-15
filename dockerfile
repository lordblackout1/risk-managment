# Use official lightweight Python image
FROM python:3.11-bookworm
# Use a slim version of Python 3.11 for a smaller image size
# Use the official Python 3.11 image based on Debian Bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y libpq-dev gcc && apt-get clean

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8501
# Command to run the Streamlit app
CMD ["streamlit", "run", "risk_analysis_app.py"]
