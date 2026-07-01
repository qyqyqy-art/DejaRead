docker run \
    --rm \
    --gpus all \
    --network host \
    -v /home/yqiu/PaddleOCR-VL-1.6:/home/yqiu/PaddleOCR-VL-1.6 \
    -e CUDA_VISIBLE_DEVICES=5 \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --model_dir /home/yqiu/PaddleOCR-VL-1.6/PaddlePaddle/PaddleOCR-VL-1___6 --host 0.0.0.0 --port 8080 --backend vllm