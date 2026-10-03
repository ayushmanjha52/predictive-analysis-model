# Container for Hugging Face Spaces (Docker SDK) -- also runs anywhere
# Docker does (Koyeb, Cloud Run, a plant server, ...).
# Python version matches the one the committed models were trained with.
FROM python:3.13.1-slim

# Spaces run the container as uid 1000; give that user the app dir so
# /log_event and shadow-mode can write their CSV files.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Spaces route traffic to 7860; elsewhere set PORT to override.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
