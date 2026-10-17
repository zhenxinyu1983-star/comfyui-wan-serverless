# RunPod Serverless worker for ComfyUI video generation.
# Base: hearmeman template (same environment as the tested pod).
# Models are NOT baked in — they come from the network volume at /runpod-volume.
FROM hearmeman/comfyui-wan-template:v29

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# Handler + startup script
COPY rp_handler.py /rp_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ComfyUI API port (for reference; serverless uses the handler, not direct HTTP)
EXPOSE 8188

CMD ["/start.sh"]
