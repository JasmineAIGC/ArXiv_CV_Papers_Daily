"""DeepSeek 大模型 API 客户端封装（OpenAI 兼容接口）"""
import time

import requests

from typing import Optional, List, Dict
from dataclasses import dataclass

# 复用连接池，避免每次请求都重新建立TCP连接
_SESSION = requests.Session()


@dataclass
class Message:
    """消息内容"""
    content: str


@dataclass
class Choice:
    """响应选项"""
    message: Message


@dataclass
class DeepSeekResponse:
    """DeepSeek API 响应封装，兼容 OpenAI SDK 格式"""
    choices: List[Choice]

    @classmethod
    def from_api_response(cls, response_data: dict) -> "DeepSeekResponse":
        """从 API 响应数据创建 DeepSeekResponse 对象"""
        try:
            choices = response_data.get("choices", [])
            parsed_choices = []
            for choice in choices:
                message = choice.get("message", {})
                content = message.get("content", "")
                parsed_choices.append(Choice(message=Message(content=str(content).strip())))
            return cls(choices=parsed_choices)
        except Exception as e:
            raise ValueError(f"解析 DeepSeek API 响应失败: {e}, 响应数据: {response_data}")


class DeepSeekCompletions:
    """DeepSeek Chat Completions 接口封装"""

    def __init__(self, client: "DeepSeekClient"):
        self.client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        thinking: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> DeepSeekResponse:
        """创建对话完成请求（兼容 OpenAI chat/completions 接口）"""
        normalized_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                normalized_messages.append({"role": role, "content": content})
            else:
                normalized_messages.append({"role": role, "content": str(content)})

        payload = {
            "model": model,
            "messages": normalized_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        # 添加 thinking 参数（如果提供）：默认开启会消耗大量 token，翻译/分类等任务需关闭
        if thinking is not None:
            payload["thinking"] = thinking

        # 发送请求
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json"
        }

        # 增强请求的重试机制，避免偶发网络错误
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 2)

        for attempt in range(max_retries):
            try:
                response = _SESSION.post(
                    f"{self.client.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=kwargs.get("timeout", 60)
                )

                if response.status_code != 200:
                    raise Exception(f"DeepSeek API 请求失败: {response.status_code} - {response.text}")

                response_data = response.json()
                return DeepSeekResponse.from_api_response(response_data)
            except Exception as e:
                if attempt < max_retries - 1:
                    # 指数退避，避免瞬时网络抖动导致长时间阻塞
                    time.sleep(min(retry_delay * (2 ** attempt), 30))
                    continue
                raise


class DeepSeekChat:
    """DeepSeek Chat 接口封装"""

    def __init__(self, client: "DeepSeekClient"):
        self.completions = DeepSeekCompletions(client)


class DeepSeekClient:
    """DeepSeek 大模型客户端

    提供与 OpenAI SDK 兼容的接口，方便替换现有的豆包/ChatGLM 调用。

    使用示例:
        client = DeepSeekClient(api_key="your-api-key")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "你好"}],
            temperature=0.7,
            max_tokens=100
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com"
    ):
        """初始化 DeepSeek 客户端

        Args:
            api_key: API 密钥
            model: 模型名称，默认为 deepseek-chat
            base_url: API 基础 URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.chat = DeepSeekChat(self)
