# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import copy
from typing import Optional
import requests
import time
import json
import re

from utils import (
    OutOfQuotaException,
    AccessTerminatedException,
    write_json,
    track_usage
)


class LlmInterface(object):
    _META_JPEG = "image/jpeg"
    _META_PNG = "image/png"
    _META_GIF = "image/gif"
    _META_WEBP = "image/webp"

    def __init__(self,
                 model: str, 
                 endpoints: str, 
                 api_key: str,
                 temperature: float = 0.0,
                 n: int = 1,
                 timeout: float = 600.0,
                 max_retry: int = 5,
                 sleep_sec: float = 5.0,
                 usage_tracking_path: Optional[str] = None) -> None:
        self.model = model
        self.endpoints = endpoints
        self.api_key = api_key
        self.temperature = temperature
        self.n = n
        self.timeout = timeout
        self.max_retry = max_retry
        self.sleep_sec = sleep_sec
        self.usage_tracking_path = usage_tracking_path
    
    def add_prompt(self,
                   text_prompt: str, 
                   images: list[str],
                   fmt: list[str],
                   role: str,
                   chat_history: list = []) -> list[list[str, list[dict]]]:
        new_chat_history = copy.deepcopy(chat_history)
        content = [
            {
                "type": "text", 
                "text": text_prompt
            }
        ]
        if images:
            for single_image_info, single_fmt in zip(images, fmt):
                if single_fmt.upper() in ["JPG", "JPEG"]:
                    meta_data = self._META_JPEG
                elif single_fmt.upper() == "PNG":
                    meta_data = self._META_PNG
                elif single_fmt.upper() == "GIF":
                    meta_data = self._META_GIF
                elif single_fmt.upper() == "WEBP":
                    meta_data = self._META_WEBP
                else:
                    raise ValueError(f"Unsupported image format: {single_fmt}. Supported formats are jpg, jpeg, png, gif, webp.")
                
                content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:{meta_data};base64,{single_image_info}"
                    }
                })
        
        new_chat_history.append([
            role,
            content
        ])

        return new_chat_history
    
    def infer(self,
              chat_history: list[list[str, list[dict]]],
              extra_headers: str = "") -> str:
        if extra_headers:
            headers = {
                "Content-Type": "application/json",
                extra_headers: self.api_key
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

        data = {
            "model": self.model,
            "messages": [],
            'temperature': self.temperature,
            'n': self.n
        }

        # claude official
        if "claude" in self.model and "https://api.anthropic.com" in self.endpoints:
            # use claude official headers
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            for role, content in chat_history:
                if role == "system":
                    data['system'] = content[0]['text']
                else:
                    converted_content = []
                    for item in content:
                        if item['type'] == "text":
                            converted_content.append({
                                "type": "text", 
                                "text": item['text']
                            })
                        elif item['type'] == "image_url":
                            # url
                            if item['image_url']['url'].startswith("http"):
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": item['image_url']['url']
                                    }
                                })
                            # base64
                            else:
                                # extract media_type
                                match = re.match(r"data:(image/\w+);base64,", item['image_url']['url'])
                                media_type = match.group(1) 
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": item['image_url']['url'].replace(f"data:{media_type};base64,", "")
                                    }
                                })
                        else:
                            raise ValueError(f"Invalid content type: {item['type']}")
                    data["messages"].append({
                        "role": role, 
                        "content": converted_content
                    })
        
        # general
        else:
            for role, content in chat_history:
                data["messages"].append({
                    "role": role, 
                    "content": content
                })
        
        max_retry = self.max_retry
        while max_retry > 0:
            try:
                if "claude" in self.model and "https://api.anthropic.com" in self.endpoints:
                    start_time = time.time()
                    res = requests.post(self.endpoints, headers=headers, data=json.dumps(data), timeout=self.timeout)
                    end_time = time.time()
                    print(f"chat with llm time: {end_time - start_time:.4f}s")
                    res.raise_for_status()
                    res_json = res.json()
                    res_content = res_json['content'][0]['text']
                else:
                    start_time = time.time()
                    res = requests.post(self.endpoints, headers=headers, json=data, timeout=self.timeout)
                    end_time = time.time()
                    print(f"chat with llm time: {end_time - start_time:.4f}s")
                    res.raise_for_status()
                    res_json = res.json()
                    res_content = res_json['choices'][0]['message']['content']
                
                print(f"current llm response is as follows: {res_content}")

                if self.usage_tracking_path:
                    usage = track_usage(res_json)
                    write_json(self.usage_tracking_path, usage, json_type="list", mode='a')
                    print(f"record current chat tokens to {self.usage_tracking_path}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if "You exceeded your current quota, please check your plan and billing details" in e.response.text:
                        raise OutOfQuotaException(self.api_key)
                    elif "Your access was terminated due to violation of our policies" in e.response.text:
                        raise AccessTerminatedException(self.api_key)
                    else:
                        print(f"Rate Limit Exceeded: {e.response.text}")
                else:
                    print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Network Error: {e}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
            except KeyError as e:
                print(f"Missing Key in Response: {e}")
            except Exception as e:
                print(f"Unexpected Error: {e}")
            else:
                # success break
                break

            print(f"Sleep {self.sleep_sec} before retry...")
            time.sleep(self.sleep_sec)
            max_retry -= 1

        else:
            print(f"Failed after {max_retry} retries...")
            raise RuntimeError("unable to connect to endpoints")

        return res_content
