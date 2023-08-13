# 医药推荐系统

该项目是一个对话式医药推荐系统，利用langchain, GPT-3.5 Turbo和医疗知识索引，基于症状提供适当的药品推荐。同时，还包括了对一般查询的回复，比如"关于我们"和"联系管理员"。

## 安装

1. 将本仓库克隆到您的本地计算机：

    ```bash
    git clone https://github.com/KANGRUIMING/SZU_medcabinet_demo.git
    ```

2. 进入项目目录：

    ```bash
    cd SZU_medcabinet_demo
    ```

3. 使用`pip`安装所需的依赖项：

    ```bash
    pip install -r requirements.txt
    ```

## 配置

1. 重命名`constants.example.py`文件为 `constants.py` ，将`APIKEY`替换为您的OpenAI API密钥。

## 使用方法

1. 使用以下命令运行主要脚本：

    ```bash
    python app.py
    ```

2. 这将启动一个Web界面，您可以在其中输入您的症状，系统会生成相应的药品推荐。
3. 如果您需要更改数据库，确保将您的数据放置在`data/`目录下。请注意，私人数据可能包含敏感信息，因此请务必采取适当的安全措施。可以使用 `.txt`, `.pdf`等格式。

## 界面

Web界面提供一个文本框，您可以在其中输入症状。AI将根据您提供的症状生成药品推荐。

## 其他查询

系统还会对以下其他查询做出回复：

- "关于我们"：提供关于团队或组织的信息。
- "联系管理员"：提供在紧急情况下的联系信息。

## 示例

以下是您可以在界面中使用的一些示例查询：

- 查询："关于我们"
- 查询："联系管理员"

## 注意事项

- 在紧急情况下，请务必寻求及时的医疗帮助。
- 回复时间可能会受网络连接情况影响。

---


