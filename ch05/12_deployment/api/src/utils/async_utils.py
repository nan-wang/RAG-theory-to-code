"""异步工具函数：提供带重试机制的异步装饰器。"""

import asyncio


def async_retry(max_retries: int = 3, delay: int = 1):
    """返回一个装饰器，使异步函数在失败时自动重试指定次数。"""
    def decorator(func):
        """包裹目标异步函数，添加重试逻辑。"""
        async def wrapper(*args, **kwargs):
            """执行函数，每次失败后等待 delay 秒后重试，超出次数则抛出异常。"""
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    print(f"Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)

            raise ValueError(f"Failed after {max_retries} attempts")

        return wrapper

    return decorator
