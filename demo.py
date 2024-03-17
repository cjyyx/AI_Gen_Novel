from AIGN import AIGN
from ideas import idea_list
from LLM import chatLLM

aign = AIGN(chatLLM)

user_idea = idea_list[1]
user_requriments = "主角独自一人行动。非常重要！主角不要有朋友！！！"
# embellishment_idea="""
# 请使用文言文创作
# """
# embellishment_idea = """
# - 使用发癫文学的风格
# - 在正文中添加表情包：😂😅😘💕😍👍
# """

aign.genNovelOutline(user_idea)
aign.genBeginning(user_requriments)

while 1:
    aign.genNextParagraph()
