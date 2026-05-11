#!/usr/bin/env python3
"""
使用 vLLM 部署 Qwen3-14B 并通过 OpenAI 兼容 API 调用，
同时将每次请求的输入、输出记录到日志文件。

依赖：
    pip install vllm openai

前提：
    - 模型 Qwen3-14B 已经下载到本地或可访问的路径（默认使用 "Qwen/Qwen3-14B"，会自动从 HuggingFace 下载）
    - GPU 显存足够（14B 模型大约需要 28GB+，建议使用 A100 或量化版本）
"""

import subprocess
import time
import logging
import json
import sys
from pathlib import Path
from openai import OpenAI


# ================== 配置 ==================
MODEL_NAME = "Qwen/Qwen3-14B"          # vLLM 支持的模型标识，也可以是本地路径
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
API_BASE = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
LOG_FILE = "vllm_qwen_calls.log"

# 测试对话消息（可根据需要修改）
TEST_MESSAGES = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "请用一句话介绍一下深度学习。"}
]

# vLLM 启动参数（可根据硬件调整）
VLLM_ARGS = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_NAME,
    "--host", VLLM_HOST,
    "--port", str(VLLM_PORT),
    "--trust-remote-code",              # Qwen 模型需要
    "--dtype", "auto",                  # 自动选择精度
    "--max-model-len", "4096",          # 限制最大长度，减少显存占用
    # 可选量化参数，例如使用 GPTQ/AWQ 时可取消注释：
    # "--quantization", "awq",
]
# =========================================


def setup_logger(log_file: str) -> logging.Logger:
    """配置日志器：同时输出到文件和终端"""
    logger = logging.getLogger("vllm_api_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 终端输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def start_vllm_server(vllm_args: list, logger: logging.Logger) -> subprocess.Popen:
    logger.info("正在启动 vLLM 服务器...")
    logger.info(f"启动命令: {' '.join(vllm_args)}")
    try:
        proc = subprocess.Popen(
            vllm_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        # 启动一个线程实时打印服务器日志（可选）
        import threading
        def log_output():
            for line in proc.stdout:
                if line:
                    logger.info(f"[vLLM] {line.rstrip()}")
        threading.Thread(target=log_output, daemon=True).start()
        return proc
    except Exception as e:
        logger.error(f"启动 vLLM 服务器失败: {e}")
        sys.exit(1)


def wait_for_server(api_base: str, logger: logging.Logger, timeout: int = 600):
    """等待 vLLM 服务器就绪（通过 /v1/models 端点）"""
    import requests
    start_time = time.time()
    url = f"{api_base}/models"
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("vLLM 服务器已就绪。")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    logger.error("等待 vLLM 服务器超时，退出。")
    sys.exit(1)


def test_call_and_log(api_base: str, logger: logging.Logger, messages: list):
    """使用 openai 库调用 API 并记录日志"""
    client = OpenAI(
        base_url=api_base,
        api_key="not-needed"   # vLLM 默认不校验 key，随便填
    )

    logger.info("=" * 50)
    logger.info("开始测试调用...")

    # 记录输入
    input_info = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256
    }
    logger.info(f"请求输入: {json.dumps(input_info, ensure_ascii=False, indent=2)}")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )
        # 提取回复内容
        output_text = completion.choices[0].message.content
        # 记录完整输出对象（也可只记录文本）
        output_info = {
            "id": completion.id,
            "model": completion.model,
            "usage": completion.usage.dict() if completion.usage else None,
            "content": output_text
        }
        logger.info(f"请求输出: {json.dumps(output_info, ensure_ascii=False, indent=2)}")
        logger.info("调用成功。")
    except Exception as e:
        logger.error(f"调用失败: {e}")
        logger.info("=" * 50)


def main():
    logger = setup_logger(LOG_FILE)
    logger.info("=== vLLM + Qwen3-14B API 测试脚本启动 ===")

    # 1. 启动服务
    server_proc = start_vllm_server(VLLM_ARGS, logger)

    try:
        # 2. 等待服务就绪
        wait_for_server(API_BASE, logger)

        # 3. 发起调用并记录日志（可以循环多次）
        test_call_and_log(API_BASE, logger, TEST_MESSAGES)

        # 如果需要更多测试，可以在此添加更多调用
        # test_call_and_log(...)

    finally:
        # 4. 清理：关闭服务器进程
        logger.info("正在关闭 vLLM 服务器...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        logger.info("vLLM 服务器已关闭。")
        logger.info("=== 脚本结束 ===")


if __name__ == "__main__":
    main()