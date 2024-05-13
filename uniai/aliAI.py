import os
import random
from http import HTTPStatus

import dashscope


def aliChatLLM(model_name, api_key=None):
    """
    model_name 取值
    - qwen1.5-7b-chat
    - qwen1.5-14b-chat
    - qwen1.5-72b-chat
    - qwen-turbo
    - qwen-max
    """
    api_key = os.environ.get("ALI_AI_API_KEY", api_key)

    def chatLLM(
        messages: list,
        temperature=0.85,
        top_p=0.8,
        stream=False,
    ) -> dict:
        if not stream:
            response = dashscope.Generation.call(
                model=model_name,
                api_key=api_key,
                messages=messages,
                seed=random.randint(1, 10000),
                temperature=temperature,
                top_p=top_p,
                result_format="message",
            )
            if response.status_code == HTTPStatus.OK:
                return {
                    "content": response.output.choices[0].message.content,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }
            else:
                error_info = (
                    "Request id: %s, Status code: %s, error code: %s, error message: %s"
                    % (
                        response.request_id,
                        response.status_code,
                        response.code,
                        response.message,
                    )
                )
                raise ValueError(f"Error in response: {error_info}")
        else:
            responses = dashscope.Generation.call(
                model=model_name,
                api_key=api_key,
                messages=messages,
                seed=random.randint(1, 10000),
                temperature=temperature,
                top_p=top_p,
                result_format="message",
                stream=True,
                output_in_full=True,  # get streaming output incrementally
            )

            def respGenerator():
                for response in responses:
                    if response.status_code == HTTPStatus.OK:
                        yield {
                            "content": response.output.choices[0]["message"]["content"],
                            "total_tokens": response.usage.input_tokens
                            + response.usage.output_tokens,
                        }
                    else:
                        error_info = (
                            "Request id: %s, Status code: %s, error code: %s, error message: %s"
                            % (
                                response.request_id,
                                response.status_code,
                                response.code,
                                response.message,
                            )
                        )
                        raise ValueError(f"Error in response: {error_info}")

            return respGenerator()
        
    return chatLLM
