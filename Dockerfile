FROM pytorch/pytorch:latest

# Install additional packages
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/username/project.git

# Set the working directory
WORKDIR /project

# Install the required packages
RUN pip install -r requirements.txt

# Copy the source code
COPY . .

# Expose the required port
EXPOSE 8000

# Start the application
CMD ["python", "app.py"]
