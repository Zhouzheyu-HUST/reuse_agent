<h1 align="center">
    HUST_agent
</h1>

<p align="center">
  <br>
  <b>作者: </b>
  <a href="#">周喆宇 金朔平 汪帆 张初波 刘柯佟</a>
  <br>
  <b>指导老师: </b>
  <a href="#">何强</a>
  <br>
  <b>联系方式: </b>
  <a href="mailto:[您的邮箱]">d202581827@hust.edu.cn</a>
</p>

---

## 版权声明

该项目所有权归开发者所有，未经授权，严禁通过任何媒介复制、分发或使用本项目。

## 项目介绍

本项目旨在构建一个基于复用知识库的移动端 Agent 系统，通过积累历史任务流来加速任务执行。

## 安装

### 1. 克隆仓库的目标分支

```bash
git clone https://github.com/Zhouzheyu-HUST/HUST_agent.git
```

### 2. 创建虚拟环境（可选）

```bash
conda create -n [环境名] python=3.12
conda activate [环境名]
```

### 3. 安装项目相关的依赖包

根据华为选用的模型可能需要自行增加安装的包。

```bash
pip install -r requirements_hust.txt
```

## 用法

### 1. 通过USB连接上手机

### 2. 配置 API

在 `custom/api_settings.json` 替换成自己的API

```json
{    
    "llm_comment": "这是除了agent以外用到的大模型",
    "llm_model": "Qwen3-VL-8B-Instruct",
    "llm_endpoints": ,
    "llm_api_key": ,
    "llm_check_way": ,

    "agent_comment": "这是agent的大模型",
    "agent_model": "Qwen3-VL-8B-Instruct",
    "agent_endpoints": ,
    "agent_api_key": ,
    "agent_check_way": ,

    "reflect_comment": "这是reflect模型",
    "reflect_model": "Qwen3-VL-4B-Instruct",
    "reflect_endpoints": ,
    "reflect_api_key": ,
    "reflect_check_way": 
}
```

### 3. 开启模型服务并测试

开启华为服务器上的模型，运行测试代码检查模型服务是否正常

```bash
python test_connection.py
```

### 4. 准备构建用数据集


运行`run_agent.sh`构建数据集`result`(在没有构建复用库之前默认为“不使用缓存”模式)

```bash
sh run_agent.sh 
```

### 5. 运行构建脚本，构建复用库 database

**重要提示**：构建过程中需要一直连接手机，最好从第四步开始就一直保持手机连接，因为save_workflow是读取result中的所有内容，如若每次运行出现意外中断，重新运行前需要保证result中内容完整.

```bash
sh run_save_workflow.sh
```

### 6. 确保所有任务都被编码（可选）

理论上save_workflow已经完成了这一步，如果不放心可以再运行一次

```bash
python encode_npy.py
```

### 7. 构建完成，开始测评

## 项目架构说明

### `database`

复用知识库（构建后出现）。

### `configs/api_settings`

OCR、大模型和 Agent 模型的 API。

### `custom`

我们的自定义功能。

### `gte`

一个用于文本编码的小模型。

### `Provider/gui_agent`

我们实现的 GUI Agent。

### `test_connection.py`

检查模型的 API 是否正常。

### `convert_history.py`

转化 history。

### `save_workflow.py`

转化工作流到复用知识库中。
