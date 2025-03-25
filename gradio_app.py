import torch
from threading import Thread

import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from datetime import datetime

model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

title="Chat with DeepSeek Model"

DEFAULT_SYSTEM = "You are a helpful assistant.Always enclose latex snippets with dollar signs! For example, $$\phi$$"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


@spaces.GPU
def stream_chat(message: str, history: list, system: str, temperature: float, max_new_tokens: int):
    msg_id = int(datetime.now().timestamp() * 1000)
    print(msg_id)
    conversation = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
    #for h in history:
    #    conversation.extend([{"role": h['role'], "content": h['content']}])

    conversation.append({"role": "user", "content": message})
    print(conversation)

    inputs = tokenizer.apply_chat_template(conversation,
            return_tensors="pt",
            ).to(model.device)
    print(f"inputs: {type(inputs)}")
    streamer = TextIteratorStreamer(tokenizer, timeout=6*10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=inputs,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        #max_new_tokens=max_new_tokens,
        max_new_tokens=16384,
        temperature=temperature,
        do_sample=True,
    )
    if temperature == 0:
        generate_kwargs["do_sample"] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    history.append(dict(role='assistant',
                    content='',
                    metadata={
                        'title':'💭',
                        'log':f'正在思考{message}',
                        'status':'pending',
                        }))

    state = 1 # 1 means 'thinking' and 2 means 'answering'

    for new_token in streamer:
        if new_token == '<think>':
            state = 1
        if new_token.find('</think>') != -1:
            history[-1]['metadata']['log'] = ''
            history[-1]['metadata']['status'] = 'done'
            history[-1]['metadata']['parent_id'] = msg_id
            #history.append({'role':'system',
            #                'content':DEFAULT_SYSTEM}
            #                )

            history.append(dict(role='assistant',
                                content='',
                                metadata={'id':msg_id}))
            state = 2
            yield history[-2:]
        else:
            pass
        if state == 1:
            print(new_token, end='')
            history[-1]['content'] += new_token
            yield history[-1:]
        elif state == 2:
            #print(new_token, end='')
            history[-1]['content'] += new_token
            yield history[-2:]
            #yield history[-1]['content']
        else:
            continue

chatbot = gr.Chatbot(type='messages')

with gr.Blocks(
        title="如何将CUDA环境下的AI模型平滑迁移至MUSA软件栈",
        ) as demo:
    gr.ChatInterface(
        title="如何将CUDA环境下的AI模型平滑迁移至MUSA软件栈",
        description='''摩尔线程（Moore Threads）推出的全功能GPU和MUSA软件栈为开发者提供了强大的支持，凭借其出色的兼容性和简便的迁移方式， 开发者可以实现无缝过渡。以下是将一个在CUDA环境下运行的AI模型（DeepSeek R1 Distill Qwen 1.5B）平滑迁移至MUSA软件栈的具体步骤：
#### 步骤 1：加入摩尔线程 Torch MUSA 插件
1. 首先，确认你的开发环境中安装了最新版本的摩尔线程 Torch MUSA 插件。可以访问摩尔线程的官方网站或其GitHub页面，下载并安装插件。
2. 安装完成后，使用`pip`或其他包管理工具确认插件已成功集成到你的Python环境中。
#### 步骤 2：替换已有代码中的CUDA字样为MUSA
1. 打开DeepSeek R1 Distill Qwen 1.5B的源代码，逐一搜索所有涉及到CUDA的相关调用（如`torch.cuda`、`cudaMalloc`等）。
2. 将这些CUDA函数的调用替换为MUSA相应的函数或方法，如将`cuda`替换为`musa`。
3. 确保所有数据传输、张量操作和设备管理都已正确更新，确保与MUSA软件栈的兼容性。
#### 步骤 3：运行示例
1. 在完成所有修改后，对代码进行全面检查，确保没有遗漏任何CUDA相关的内容。
2. 运行修改后的模型，观察是否能够成功在MUSA环境中启动和执行。如果遇到任何错误，请仔细阅读错误信息，排查兼容性问题。
3. 最后，通过对比在CUDA和MUSA环境下的执行结果和性能，确保迁移的成功。
通过遵循上述步骤，开发者可以将DeepSeek R1 Distill Qwen 1.5B模型平滑地迁移至MUSA软件栈，从而充分利用摩尔线程的全功能GPU带来的性能优势。这样的迁移不仅能提高开发效率，还能有效降低技术转型的成本。''',
        type='messages',
        fn=stream_chat,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Text(
                value="",
                label="System",
                render=False,
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.8,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=4096,
                step=1,
                value=1024,
                label="Max new tokens",
                render=False,
            ),
        ],
        examples=[
            ["解方程：3x + 5 = 20。求 x 的值。", ""],
            ["一个矩形的长是8厘米，宽是5厘米。请计算该矩形的周长和面积。", ""],
            ["计算圆锥绕其底面直径旋转 (180^\circ) 形成的立体体积。", ""],
            ["设函数 f(x) = 2x + 3。求 f(4) 的值。", ""],
            ["一个袋子里有5个红球和3个蓝球。如果随机抽取一个球，抽到红球的概率是多少？", ""],
            ["∫(x^2 + 2x) dx，区间为从1到4。请给出结果，并解释你的过程。", ""],
            ["用C++实现KMP算法，并加上中文注释", ""],
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(server_name = '0.0.0.0')
