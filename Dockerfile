# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /code

# Copy requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files (including dataset, chroma_db, and backend code)
COPY . /code

# Set environment variables (Hugging Face Spaces runs on port 7860 by default)
ENV PORT=7860

# Command to run uvicorn on Hugging Face Space's default port 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
