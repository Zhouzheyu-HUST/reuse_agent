import os
import json
import time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
from typing import Union

# ================== 本地 API 配置区 ==================

def load_api_config(filename="api_settings.json"):
    """
    从当前脚本所在目录向上递归查找 custom/api_settings.json
    (与 qwen_wrapper.py 保持完全一致)
    """
    current_path = Path(__file__).resolve().parent

    for parent in [current_path] + list(current_path.parents):
        config_file = parent / "custom" / filename
        if config_file.exists():
            try:
                return json.loads(config_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
    return {}

def compress_base64_image(base64_str: str, factor: float = 0.5, quality: int = 80) -> str:
    """
    將 Base64 圖片解碼、壓縮（縮放 + 降低畫質），再重新編碼為 Base64
    :param base64_str: 原始 Base64 字串
    :param factor: 縮放比例 (例如 0.5 表示長寬各縮小一半)
    :param quality: JPEG 壓縮品質 (0-100，預設 80)
    :return: 壓縮後的 Base64 字串
    """
    try:
        # [速度優化 1] 使用 find 和切片取代 split，降低記憶體開銷並加快字串處理
        idx = base64_str.find(',')
        if idx != -1:
            base64_str = base64_str[idx+1:]

        # 1. 將 Base64 解碼為二進制數據
        img_data = base64.b64decode(base64_str)
        
        # 2. 使用 PIL 打開圖片
        img = Image.open(BytesIO(img_data))
        
        # 3. 轉換為 RGB 模式 (防止 PNG 的透明通道在存成 JPEG 時報錯)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # 4. 根據 factor 計算新尺寸並等比例縮放
        if factor > 0 and factor != 1.0:
            new_width = max(1, int(img.width * factor))
            new_height = max(1, int(img.height * factor))
            # [速度優化 2] 將 LANCZOS 替換為 BILINEAR，大幅提升縮放速度
            img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        # 5. 將壓縮後的圖片存入內存緩衝區，格式轉為 JPEG 以獲得更好的壓縮率
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        
        # 6. 重新編碼為 Base64
        compressed_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return compressed_base64
    except Exception as e:
        print(f"[Image Compression Error] {e}")
        # 如果壓縮失敗，退回原始的 Base64
        return base64_str
    
def fmt_point(x, y, width, height):
    w = width if width > 0 else 1276
    h = height if height > 0 else 2848
    return f"[{x}, {y}] (relative: {x/w:.2f}, {y/h:.2f})"

class ReflectWrapper:
    def __init__(self, max_retry=3, retry_waiting_seconds=2):
        # 1. 加载配置
        config = load_api_config()
        
        # 2. 提取配置项 (请确保 api_settings.json 中有这些键，或者提供默认值)
        # 注意：这里默认端口写了 8000，你需要根据你的本地部署情况调整
        self.api_url = config.get("SmolVLM_endpoints")
        self.api_key = config.get("SmolVLM_api_key") 
        self.model = config.get("SmolVLM_model") # 替换为你部署的模型实际名称
        
        self.max_retry = max_retry
        self.retry_waiting_seconds = retry_waiting_seconds

    def predict_reflection(self, 
                           query: str, 
                           pre_image_base64: str, 
                           next_image_base64: str, 
                           action_seq: list,
                           thought: Union[str, None],
                           width: int,
                           height: int) -> str:
        """
        判断动作执行是否有效。
        """
        factor = 0.25
        quality = 80
        start_time = time.time()
        compressed_pre_image = compress_base64_image(pre_image_base64, factor, quality)
        compressed_next_image = compress_base64_image(next_image_base64, factor, quality)
        #compressed_time = time.time() - start_time
        #print(f"compressed_time: {compressed_time:.2f} seconds. No action sequence to reflect on, skipping reflection.", stdout=True)
        
        # --- 1. 动作解析与坐标归一化 ---
        action_desc_list = []
        
        for action in action_seq:
            act_type = action.get('type')
            params = action.get('params', {})
            
            try:
                if act_type in ['click', 'longclick']:
                    pts = params.get('points', [])
                    # 安全檢查：確保座標陣列至少有 2 個元素
                    if len(pts) >= 2:
                        desc = f"{'long click' if act_type == 'longclick' else 'click'} at {fmt_point(pts[0], pts[1], width, height)}"
                    else:
                        desc = f"{act_type} at unknown points"
                    
                elif act_type == 'set_text':
                    text = params.get('text', '')
                    desc = f"input text '{text}'"
                    
                elif act_type == 'scroll':
                    pts = params.get('points', [])
                    # 安全檢查：滑動需要 4 個座標值 (起點和終點)
                    if len(pts) >= 4:
                        start = fmt_point(pts[0], pts[1], width, height)
                        end = fmt_point(pts[2], pts[3], width, height)
                        desc = f"scroll from {start} to {end}"
                    else:
                        desc = "scroll on screen"
                    
                elif act_type in ['back', 'home', 'enter']:
                    desc = f"press {act_type.upper()} button"
                    
                else:
                    desc = f"perform action '{act_type}'"
                    
                action_desc_list.append(desc)
                
            except Exception as e:
                print(f"[ReflectWrapper] Parsing Error for action {action}: {e}")
                action_desc_list.append(f"perform action '{act_type}'")

        action_str = " then ".join(action_desc_list)

        # --- 2. 构建 Prompt (使用标准 OpenAI 多模态格式) ---
        system_prompt = (
            "You are a mobile agent critic. Your task is to evaluate if the agent's action "
            "successfully caused the expected change on the screen based on the User's Query.\n"
            f"Screen Resolution: {width}x{height}.\n"
            "Compare the 'Before' and 'After' screenshots carefully.\n"
            "Output '1' if the action was successful or valid. Do not output any other words just output '1'\n"
            "Output a brief reason less than 10 words if the action failed (e.g., 'Page did not load', 'Button click missed')."
        )

        text_content = f"User Query: {query}\n\n"
        
        # 严格判断：只有 thought 不是 None 且去空白后不是空字符串，才加上这一段
        if thought is not None and str(thought).strip():
            text_content += f"Agent's Intended Thought: {thought}\n\n"
            
        text_content += f"Action Performed: {action_str}\n\nScreenshot Before Action:"

        messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": text_content
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{compressed_pre_image}"}
                    },
                    {
                        "type": "text", 
                        "text": "\nScreenshot After Action:"
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{compressed_next_image}"}
                    },
                    {
                        "type": "text", 
                        "text": "\nDid the action work? (Output '1' or Reason)"
                    }
                ]
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # 反思任务通常需要确定性，温度调低
            "max_tokens": 512,
            "stream": True  # 这是应该是True还是false好一些呢
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # --- 3. 发送 HTTP 请求并增加重试机制 ---
        counter = self.max_retry
        wait_seconds = self.retry_waiting_seconds

        while counter > 0:
            try:
                # [修改] 增加 stream=True 参数，让 requests 开启流式接收
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30,  
                    stream=True  
                )

                if response.ok:
                    full_content = ""
                    # 逐行读取流式数据
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                # 检查流是否结束
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        # 兼容标准的 delta 结构和某些非标 API 的 message 结构
                                        delta = chunk["choices"][0].get("delta", chunk["choices"][0].get("message", {}))
                                        if "content" in delta:
                                            full_content += delta["content"]
                                            
                                            # [核心逻辑：提前终止]
                                            stripped_content = full_content.strip()
                                            if stripped_content:
                                                # 如果第一个有效字符是 '1'
                                                if stripped_content.startswith("1"):
                                                    # 立即强行切断与服务器的连接，停止生成！
                                                    response.close()
                                                    return "1"
                                                # 如果不是 '1' (说明是失败原因)，则什么都不做，继续接收完整的生成结果
                                except json.JSONDecodeError:
                                    pass
                    
                    # 循环自然结束（说明是以文字开头的失败原因），返回完整拼接的内容
                    return full_content.strip() if full_content else "1"
                
                print(f"[ReflectWrapper] API Error: HTTP {response.status_code} - {response.text}")
                counter -= 1
                time.sleep(wait_seconds)
                wait_seconds *= 2  # 指数退避

            except requests.exceptions.RequestException as e:
                print(f"[ReflectWrapper] Network or Timeout Error: {e}")
                counter -= 1
                time.sleep(wait_seconds)
                wait_seconds *= 2
        
        # 如果所有重试都失败，默认返回 1 (通过)，避免整个 Agent 流程卡死
        print("[ReflectWrapper] Max retries reached. Defaulting to '1'.")
        return "1"