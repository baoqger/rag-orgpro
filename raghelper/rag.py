# 导入依赖
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.llms.dashscope import DashScope
# 这两行代码是用于消除 WARNING 警告信息，避免干扰阅读学习，生产环境中建议根据需要来设置日志级别
import logging
logging.basicConfig(level=logging.ERROR)
from llama_index.llms.openai_like import OpenAILike
import os

def indexing(document_path="./docs", persist_path="knowledge_base/test"):
    """
    建立索引并持久化存储
    参数
      path(str): 文档路径
    """
    index = create_index(document_path)
    # 持久化索引，将索引保存为本地文件
    index.storage_context.persist(persist_path)

def create_index(document_path="./docs"):
    """
    建立索引
    参数
      path(str): 文档路径
    """
    # 解析 ./docs 目录下的所有文档
    documents = SimpleDirectoryReader(document_path).load_data()
    # 建立索引
    index = VectorStoreIndex.from_documents(
        documents,
        # 指定embedding 模型
        embed_model=DashScopeEmbedding(
            # 你也可以使用阿里云提供的其它embedding模型：https://help.aliyun.com/zh/model-studio/getting-started/models#3383780daf8hw
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
        )
    )
    return index

def load_index(persist_path="knowledge_base/test"):
    """
    加载索引
    参数
      persist_path(str): 索引文件路径
    返回
      VectorStoreIndex: 索引对象
    """
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    return load_index_from_storage(storage_context,embed_model=DashScopeEmbedding(
      model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
    ))

def create_query_engine(index):
    """
    创建查询引擎
    参数
      index(VectorStoreIndex): 索引对象
    返回
      QueryEngine: 查询引擎对象
    """
    
    query_engine = index.as_query_engine(
      # 设置为流式输出
      streaming=True,
      # 此处使用qwen-plus-0919模型，你也可以使用阿里云提供的其它qwen的文本生成模型：https://help.aliyun.com/zh/model-studio/getting-started/models#9f8890ce29g5u
      llm=OpenAILike(
          model="qwen-plus-0919",
          api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
          api_key=os.getenv("DASHSCOPE_API_KEY"),
          is_chat_model=True
          ))
    return query_engine

def ask(question, query_engine):
    """
    向答疑机器人提问
    参数
      question(str): 问题
      query_engine(QueryEngine): 查询引擎对象
    返回
      str: 回答
    """
    streaming_response = query_engine.query(question)
    streaming_response.print_response_stream()

def ask_with_debug(question, query_engine):
    # 更新提示模板
    update_prompt_template(query_engine=query_engine)

    # 输出问题
    print('=' * 50)  # 使用乘法生成分割线
    print(f'🤔 问题：{question}')
    print('=' * 50 + '\n')  # 使用乘法生成分割线

    # 获取回答
    response = query_engine.query(question)

    # 输出回答
    print('🤖 回答：')
    if hasattr(response, 'print_response_stream') and callable(response.print_response_stream):
        response.print_response_stream()
    else:
        print(str(response))

    # 输出参考文档
    print('\n' + '-' * 50)  # 使用乘法生成分割线
    print('📚 参考文档：\n')
    for i, source_node in enumerate(response.source_nodes, start=1):
        print(f'文档 {i}:')
        print(source_node)
        print()

    print('-' * 50)  # 使用乘法生成分割线

    return response


from llama_index.core import PromptTemplate
def update_prompt_template(
        query_engine,
        qa_prompt_tmpl_str = (
        "你叫公司小蜜，是公司的答疑机器人。你需要仔细阅读参考信息，然后回答大家提出的问题。"
        "注意事项：\n"
        "1. 根据上下文信息而非先验知识来回答问题。\n"
        "2. 如果是工具咨询类问题，请务必给出下载地址链接。\n"
        "3. 如果员工部门查询问题，请务必注意有同名员工的情况，可能有2个、3个甚至更多同名的人\n"
        "以下是参考信息。"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "问题：{query_str}\n。"
        "回答："
    )):
    """
    修改prompt模板
    输入是prompt修改前的query_engine，以及提示词模板；输出是prompt修改后的query_engine
    """
    qa_prompt_tmpl_str = qa_prompt_tmpl_str
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    # print("提示词模板修改成功")
    return query_engine
