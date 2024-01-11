from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
import os
from dotenv import load_dotenv

load_dotenv()


n_gpu_layers = int(os.getenv("N_GPU_LAYERS"))
n_ctx = int(os.getenv("N_CTX"))
n_batch = int(os.getenv("N_BATCH"))
agent_prompt = os.getenv("AGENT_PROMPT")
predefined_messages_formatter_type = os.getenv("PREDEFINED_MESSAGES_FORMATTER_TYPE")
debug_output = os.getenv("DEBUG_OUTPUT") == "True"

class Model:
    def __init__(self, model_path):
        print('Initializing model...this may take a while.')
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers,
                         n_ctx=n_ctx, n_batch=n_batch, use_mmap=False, verbose=debug_output)

        self.wrapped_model = LlamaCppAgent(self.llm, debug_output=debug_output,
                                           system_prompt=agent_prompt, predefined_messages_formatter_type=MessagesFormatterType[predefined_messages_formatter_type])

    async def generate(self, input_text, max_tokens=-1):
        output = self.wrapped_model.get_chat_response(
            input_text, add_response_to_chat_history=False, add_message_to_chat_history=False, max_tokens=max_tokens)

        return output