import random
import threading
import time

import gradio as gr

from AIGN import AIGN
from ideas import idea_list
from LLM import chatLLM

STREAM_INTERVAL = 0.2


def make_middle_chat():
    carrier = threading.Event()
    carrier.history = []

    def middle_chat(messages, temperature=None, top_p=None):
        nonlocal carrier
        carrier.history.append([None, ""])
        if len(carrier.history) > 20:
            carrier.history = carrier.history[-16:]
        try:
            for resp in chatLLM(
                messages, temperature=temperature, top_p=top_p, stream=True
            ):
                output_text = resp["content"]
                total_tokens = resp["total_tokens"]

                carrier.history[-1][1] = f"total_tokens: {total_tokens}\n{output_text}"
            return {
                "content": output_text,
                "total_tokens": total_tokens,
            }
        except Exception as e:
            carrier.history[-1][1] = f"Error: {e}"
            raise e

    return carrier, middle_chat


def gen_ouline_button_clicked(aign, user_idea, history):
    aign.user_idea = user_idea

    carrier, middle_chat = make_middle_chat()
    carrier.history = history
    aign.novel_outline_writer.chatLLM = middle_chat

    gen_ouline_thread = threading.Thread(target=aign.genNovelOutline)
    gen_ouline_thread.start()

    while gen_ouline_thread.is_alive():
        yield [
            aign,
            carrier.history,
            aign.novel_outline,
            gr.Button(visible=False),
        ]
        time.sleep(STREAM_INTERVAL)
    yield [
        aign,
        carrier.history,
        aign.novel_outline,
        gr.Button(visible=False),
    ]


def gen_beginning_button_clicked(
    aign, history, novel_outline, user_requriments, embellishment_idea
):
    aign.novel_outline = novel_outline
    aign.user_requriments = user_requriments
    aign.embellishment_idea = embellishment_idea

    carrier, middle_chat = make_middle_chat()
    carrier.history = history
    aign.novel_beginning_writer.chatLLM = middle_chat
    aign.novel_embellisher.chatLLM = middle_chat

    gen_beginning_thread = threading.Thread(target=aign.genBeginning)
    gen_beginning_thread.start()

    while gen_beginning_thread.is_alive():
        yield [
            aign,
            carrier.history,
            aign.writing_plan,
            aign.temp_setting,
            aign.novel_content,
            gr.Button(visible=False),
        ]
        time.sleep(STREAM_INTERVAL)
    yield [
        aign,
        carrier.history,
        aign.writing_plan,
        aign.temp_setting,
        aign.novel_content,
        gr.Button(visible=False),
    ]


def gen_next_paragraph_button_clicked(
    aign,
    history,
    user_idea,
    novel_outline,
    writing_memory,
    temp_setting,
    writing_plan,
    user_requriments,
    embellishment_idea,
):
    aign.user_idea = user_idea
    aign.novel_outline = novel_outline
    aign.writing_memory = writing_memory
    aign.temp_setting = temp_setting
    aign.writing_plan = writing_plan
    aign.user_requriments = user_requriments
    aign.embellishment_idea = embellishment_idea

    carrier, middle_chat = make_middle_chat()
    carrier.history = history
    aign.novel_writer.chatLLM = middle_chat
    aign.novel_embellisher.chatLLM = middle_chat
    aign.memory_maker.chatLLM = middle_chat

    gen_next_paragraph_thread = threading.Thread(target=aign.genNextParagraph)
    gen_next_paragraph_thread.start()

    while gen_next_paragraph_thread.is_alive():
        yield [
            aign,
            carrier.history,
            aign.writing_plan,
            aign.temp_setting,
            aign.writing_memory,
            aign.novel_content,
            gr.Button(visible=False),
        ]
        time.sleep(STREAM_INTERVAL)
    yield [
        aign,
        carrier.history,
        aign.writing_plan,
        aign.temp_setting,
        aign.writing_memory,
        aign.novel_content,
        gr.Button(visible=False),
    ]


with gr.Blocks() as demo:
    aign = gr.State(AIGN(chatLLM))
    with gr.Row():
        with gr.Column(scale=0):
            with gr.Tab("开始"):
                user_idea_text = gr.Textbox(
                    "罗瑜，低等魔族，性格恶劣，立志要成为魔族大帝。它单枪匹马向勇者杀去...",
                    label="想法",
                    lines=4,
                    interactive=True,
                )
                user_requriments_text = gr.Textbox(
                    "主角独自一人行动。主角不需要朋友！！！",
                    label="写作要求",
                    lines=4,
                    interactive=True,
                )
                embellishment_idea_text = gr.Textbox(
                    "语言搞笑幽默", label="润色要求", lines=4, interactive=True
                )
                gen_ouline_button = gr.Button("生成大纲")
            with gr.Tab("大纲"):
                novel_outline_text = gr.Textbox(
                    label="大纲", lines=24, interactive=True
                )
                gen_beginning_button = gr.Button("生成开头")
            with gr.Tab("状态"):
                writing_memory_text = gr.Textbox(
                    label="记忆",
                    lines=6,
                    interactive=True,
                    max_lines=8,
                )
                writing_plan_text = gr.Textbox(label="计划", lines=6, interactive=True)
                temp_setting_text = gr.Textbox(
                    label="临时设定", lines=5, interactive=True
                )
                # TODO
                # gen_next_paragraph_button = gr.Button("撤销生成")
                gen_next_paragraph_button = gr.Button("生成下一段")
                # TODO
                # auto_gen_next_checkbox = gr.Checkbox(
                #     label="自动生成下一段", checked=False, interactive=True
                # )
        with gr.Column(scale=3):
            chatBox = gr.Chatbot(height=f"80vh", label="输出")
        with gr.Column(scale=0):
            novel_content_text = gr.Textbox(
                label="小说正文", lines=32, interactive=True, show_copy_button=True
            )
            # TODO
            # download_novel_button = gr.Button("下载小说")

    gen_ouline_button.click(
        gen_ouline_button_clicked,
        [aign, user_idea_text, chatBox],
        [aign, chatBox, novel_outline_text, gen_ouline_button],
    )
    gen_beginning_button.click(
        gen_beginning_button_clicked,
        [
            aign,
            chatBox,
            novel_outline_text,
            user_requriments_text,
            embellishment_idea_text,
        ],
        [
            aign,
            chatBox,
            writing_plan_text,
            temp_setting_text,
            novel_content_text,
            gen_beginning_button,
        ],
    )
    gen_next_paragraph_button.click(
        gen_next_paragraph_button_clicked,
        [
            aign,
            chatBox,
            user_idea_text,
            novel_outline_text,
            writing_memory_text,
            temp_setting_text,
            writing_plan_text,
            user_requriments_text,
            embellishment_idea_text,
        ],
        [
            aign,
            chatBox,
            writing_plan_text,
            temp_setting_text,
            writing_memory_text,
            novel_content_text,
        ],
    )


demo.queue()
demo.launch()
