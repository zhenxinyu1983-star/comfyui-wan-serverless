# RunPod Serverless worker for ComfyUI video generation.
# Base: hearmeman template (same environment as the tested pod).
# Models are NOT baked in — they come from the network volume at /runpod-volume.
FROM hearmeman/comfyui-wan-template:v29

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# 锁 transformers 到 4.44.x: 太新的版本 wav2vec2 默认不输出 hidden_states,
# 会导致 MultiTalkWav2VecEmbeds 报 'NoneType' object is not subscriptable (K22 双人失败)
RUN pip install --no-cache-dir "transformers==4.44.2"

# Handler + startup script
COPY rp_handler.py /rp_handler.py
COPY start.sh /start.sh

# Custom nodes: WanCameraOrbitEmbedding (Fun Camera Orbit, 2026-10-20)
COPY custom_nodes/ComfyUI-TST-Orbit /ComfyUI/custom_nodes/ComfyUI-TST-Orbit
RUN chmod +x /start.sh

# ComfyUI API port (for reference; serverless uses the handler, not direct HTTP)
EXPOSE 8188

CMD ["/start.sh"]
