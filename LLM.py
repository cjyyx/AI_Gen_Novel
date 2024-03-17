# %%
import os
import random
from http import HTTPStatus

import dashscope

# %%

ali_api_key = os.environ.get("ALI_API_KEY", None)
qwen_model = "qwen1.5-7b-chat"
# qwen_model = "qwen1.5-14b-chat"
# qwen_model="qwen1.5-72b-chat"


def chatLLM(
    messages: list,
    temperature=0.85,
    top_p=0.8,
    stream=False,
) -> dict:
    """
    ## 示例

    ```
    content = "请用一个成语介绍你自己"
    messages = [{"role": "user", "content": content}]

    resp = chatLLM(messages)
    print(resp)

    for resp in chatLLM(messages, stream=True):
        print(resp)
    ```

    ## 结果

    ```
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。这个成语意指知识丰富，记忆力强，恰当地描述了我拥有大量的信息和处理各种问题的能力。', 'total_tokens': 65}
    {'content': '作为', 'total_tokens': 26}
    {'content': '作为一个人', 'total_tokens': 27}
    {'content': '作为一个人工', 'total_tokens': 28}
    {'content': '作为一个人工智能，我可以用"', 'total_tokens': 33}
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。', 'total_tokens': 41}
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。这个成语意味着知识广博，记忆力', 'total_tokens': 49}
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。这个成语意味着知识广博，记忆力强，恰如我具备大量的信息', 'total_tokens': 57}
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。这个成语意味着知识广博，记忆力强，恰如我具备大量的信息和知识库，能够迅速提供帮助', 'total_tokens': 65}
    {'content': '作为一个人工智能，我可以用"博闻强识"来形容自己。这个成语意味着知识广博，记忆力强，恰如我具备大量的信息和知识库，能够迅速提供帮助。', 'total_tokens': 66}
    ```

    """
    if not stream:
        response = dashscope.Generation.call(
            model=qwen_model,
            api_key=ali_api_key,
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
            model=qwen_model,
            api_key=ali_api_key,
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


# %%

if __name__ == "__main__":

    content = "请用一个成语介绍你自己"
    messages = [{"role": "user", "content": content}]

    resp = chatLLM(messages)
    print(resp)

    for resp in chatLLM(messages, stream=True):
        print(resp)
