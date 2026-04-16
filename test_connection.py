import json
import requests
import time
from pathlib import Path

# ================== 1. 鲁棒的配置加载逻辑 ==================
def load_api_config(filename="api_settings.json"):
    """
    从当前脚本所在目录向上递归查找 custom/api_settings.json
    """
    current_path = Path(__file__).resolve().parent
    print(f"\n[1/3] 正在寻找配置文件...")
    print(f"      当前脚本路径: {current_path}")
    
    for parent in [current_path] + list(current_path.parents):
        config_file = parent / "custom" / filename
        if config_file.exists():
            print(f"      ✅ 成功加载配置文件: {config_file}")
            try:
                return json.loads(config_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"      ❌ 配置文件解析失败: {e}")
                return {}
    
    print(f"      ❌ 未找到配置文件 {filename}！请检查 configs 目录。")
    return {}

# ================== 2. 通用测试函数 ==================
def test_endpoint(name, url, key, model_name):
    print(f"\n[2/3] 开始测试 {name} 配置...")
    
    # 1. 检查配置是否为空
    if not url or not key or not model_name:
        print(f"      ❌ {name} 配置缺失！跳过测试。")
        return

    # 2. 打印当前使用的配置 (Key 进行脱敏)
    masked_key = key[:6] + "******" + key[-4:] if len(key) > 10 else "******"
    print(f"      目标 URL  : {url}")
    print(f"      目标 Model: {model_name}")
    print(f"      使用 Key  : {masked_key}")

    # 3. 构造一个极简的测试请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello! Please reply with 'Connection Successful'."}
        ],
        "temperature": 0.01,
        "max_tokens": 20,
        "stream": False
    }

    # 4. 发起请求
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        duration = time.time() - start_time
        
        # 5. 分析结果
        if response.status_code == 200:
            data = response.json()
            content = "无内容"
            returned_model = "未知"
            
            # 提取回复内容
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
            
            # 提取服务端确认的模型名 (关键步骤)
            if "model" in data:
                returned_model = data["model"]

            print(f"      ✅ 调用成功! (耗时: {duration:.2f}s)")
            print(f"      📥 服务端响应内容: \"{content}\"")
            print(f"      🆔 服务端确认模型: {returned_model}")
            
            if returned_model == model_name:
                print(f"      ✨ 验证通过：服务端返回的模型名与配置一致。")
            else:
                print(f"      ⚠️ 注意：服务端返回的模型名 ({returned_model}) 与配置 ({model_name}) 不一致，可能是服务端做了映射。")
                
        else:
            print(f"      ❌ 调用失败 (HTTP {response.status_code})")
            print(f"      错误信息: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"      ❌ 连接错误：无法连接到 {url}，请检查服务是否启动或地址是否正确。")
    except requests.exceptions.Timeout:
        print(f"      ❌ 请求超时：服务在 15秒 内未响应。")
    except Exception as e:
        print(f"      ❌ 发生异常: {e}")

# ================== 3. 主程序 ==================
if __name__ == "__main__":
    print("="*60)
    print("        LLM & Agent 接口连通性交付测试程序")
    print("="*60)

    # 加载配置
    config = load_api_config()

    if config:
        # 测试 LLM 配置
        llm_url = config.get("llm_endpoints")
        llm_key = config.get("llm_api_key")
        llm_model = config.get("llm_model")
        test_endpoint("通用大模型 (LLM)", llm_url, llm_key, llm_model)

        print("-" * 60)

        # 测试 Agent 配置
        agent_url = config.get("agent_endpoints")
        agent_key = config.get("agent_api_key")
        agent_model = config.get("agent_model")
        test_endpoint("Agent 专用模型", agent_url, agent_key, agent_model)

        print("-" * 60)

        # 测试 Reflect 配置
        reflect_url = config.get("reflect_endpoints")
        reflect_key = config.get("reflect_api_key")
        reflect_model = config.get("reflect_model")
        test_endpoint("反思模型 (Reflect)", reflect_url, reflect_key, reflect_model)
    
    print("\n" + "="*60)
    print("测试结束。请截图此屏幕作为交付验证凭证。")
    print("="*60)