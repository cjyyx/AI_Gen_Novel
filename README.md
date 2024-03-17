# 基于 LLM 和 Multi-Agent 实现 AI 写小说

## 简介

本项目计划基于大语言模型(LLM)和多智能体(Multi-Agent)，探究AI写小说能力的边界。目前具体实现参考了 RecurrentGPT。

## 在线体验

https://modelscope.cn/studios/cjyyxn/AI_Gen_Novel/summary

## 快速上手指南

### 步骤1: 环境配置

首先，安装所需的依赖项：

```bash
pip install -r requirements.txt
```

### 步骤2: 实现LLM接口

项目依赖一个大语言模型。你需要完成`LLM.py`中的`chatLLM`函数的实现。

如果你拥有阿里DashScope的灵积模型服务api_key，可将`LLM.py`文件顶部的`ali_api_key`变量替换为你的个人api_key，即可实现`chatLLM`函数。

### 步骤3: 运行项目

- 直接运行`demo.py`，将自动创作小说，并将结果保存在`novel_record.md`文件中。

- 运行`app.py`启动一个基于gradio的应用，通过打开显示的链接，您可以体验到AI小说生成的可视化过程。

