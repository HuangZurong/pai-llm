import os
from enum import Enum
from typing import Union, cast

from .unified_model import UnifiedModel


class Model(UnifiedModel, Enum):

    def __new__(cls, value: Union["Model", str]) -> "Model":
        return cast("Model", UnifiedModel.__new__(cls, value))

    DEFAULT = os.getenv("DEFAULT_MODEL", "gpt-4o")

    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_5_PREVIEW = "gpt-4.5-preview"
    O1 = "o1"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"

    GLM_4 = "glm-4"
    GLM_4V = "glm-4v"
    GLM_4V_FLASH = "glm-4v-flash"
    GLM_4V_PLUS_0111 = "glm-4v-plus-0111"
    GLM_4_PLUS = "glm-4-plus"
    GLM_4_AIR = "glm-4-air"
    GLM_4_AIR_0111 = "glm-4-air-0111"
    GLM_4_AIRX = "glm-4-airx"
    GLM_4_LONG = "glm-4-long"
    GLM_4_FLASHX = "glm-4-flashx"
    GLM_4_FLASH = "glm-4-flash"
    GLM_ZERO_PREVIEW = "glm-zero-preview"
    GLM_3_TURBO = "glm-3-turbo"

    # Groq platform models
    GROQ_LLAMA_3_1_8B = "llama-3.1-8b-instant"
    GROQ_LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    GROQ_LLAMA_3_3_70B_PREVIEW = "llama-3.3-70b-specdec"
    GROQ_LLAMA_3_8B = "llama3-8b-8192"
    GROQ_LLAMA_3_70B = "llama3-70b-8192"
    GROQ_MIXTRAL_8_7B = "mixtral-8x7b-32768"
    GROQ_GEMMA_2_9B_IT = "gemma2-9b-it"

    # TogetherAI platform models support tool calling
    TOGETHER_LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    TOGETHER_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    TOGETHER_LLAMA_3_1_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    TOGETHER_LLAMA_3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    TOGETHER_MIXTRAL_8_7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    TOGETHER_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"

    # SambaNova Cloud platform models support tool calling
    SAMBA_LLAMA_3_1_8B = "Meta-Llama-3.1-8B-Instruct"
    SAMBA_LLAMA_3_1_70B = "Meta-Llama-3.1-70B-Instruct"
    SAMBA_LLAMA_3_1_405B = "Meta-Llama-3.1-405B-Instruct"

    # SGLang models support tool calling
    SGLANG_LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    SGLANG_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    SGLANG_LLAMA_3_1_405B = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    SGLANG_LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
    SGLANG_MIXTRAL_NEMO = "mistralai/Mistral-Nemo-Instruct-2407"
    SGLANG_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
    SGLANG_QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    SGLANG_QWEN_2_5_32B = "Qwen/Qwen2.5-32B-Instruct"
    SGLANG_QWEN_2_5_72B = "Qwen/Qwen2.5-72B-Instruct"

    STUB = "stub"

    # Legacy anthropic models
    # NOTE: anthropic legacy models only Claude 2.1 has system prompt support
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"

    # Claude3 models
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-latest"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-latest"

    # Nvidia models
    NVIDIA_NEMOTRON_340B_INSTRUCT = "nvidia/nemotron-4-340b-instruct"
    NVIDIA_NEMOTRON_340B_REWARD = "nvidia/nemotron-4-340b-reward"
    NVIDIA_YI_LARGE = "01-ai/yi-large"
    NVIDIA_MISTRAL_LARGE = "mistralai/mistral-large"
    NVIDIA_MIXTRAL_8X7B = "mistralai/mixtral-8x7b-instruct"
    NVIDIA_LLAMA3_70B = "meta/llama3-70b"
    NVIDIA_LLAMA3_1_8B_INSTRUCT = "meta/llama-3.1-8b-instruct"
    NVIDIA_LLAMA3_1_70B_INSTRUCT = "meta/llama-3.1-70b-instruct"
    NVIDIA_LLAMA3_1_405B_INSTRUCT = "meta/llama-3.1-405b-instruct"
    NVIDIA_LLAMA3_2_1B_INSTRUCT = "meta/llama-3.2-1b-instruct"
    NVIDIA_LLAMA3_2_3B_INSTRUCT = "meta/llama-3.2-3b-instruct"
    NVIDIA_LLAMA3_3_70B_INSTRUCT = "meta/llama-3.3-70b-instruct"

    # Gemini models
    GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"
    GEMINI_2_0_FLASH_THINKING = "gemini-2.0-flash-thinking-exp"
    GEMINI_2_0_PRO_EXP = "gemini-2.0-pro-exp-02-05"
    GEMINI_2_0_FLASH_LITE_PREVIEW = "gemini-2.0-flash-lite-preview-02-05"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"

    # Mistral AI models
    MISTRAL_3B = "ministral-3b-latest"
    MISTRAL_7B = "open-mistral-7b"
    MISTRAL_8B = "ministral-8b-latest"
    MISTRAL_CODESTRAL = "codestral-latest"
    MISTRAL_CODESTRAL_MAMBA = "open-codestral-mamba"
    MISTRAL_LARGE = "mistral-large-latest"
    MISTRAL_MIXTRAL_8x7B = "open-mixtral-8x7b"
    MISTRAL_MIXTRAL_8x22B = "open-mixtral-8x22b"
    MISTRAL_NEMO = "open-mistral-nemo"
    MISTRAL_PIXTRAL_12B = "pixtral-12b-2409"

    # Reka models
    REKA_CORE = "reka-core"
    REKA_FLASH = "reka-flash"
    REKA_EDGE = "reka-edge"

    # Cohere models
    COHERE_COMMAND_R_PLUS = "command-r-plus"
    COHERE_COMMAND_R = "command-r"
    COHERE_COMMAND_LIGHT = "command-light"
    COHERE_COMMAND = "command"
    COHERE_COMMAND_NIGHTLY = "command-nightly"

    # Qwen models (Aliyun)
    QWEN_MAX = "qwen-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"
    QWEN_LONG = "qwen-long"
    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
    QWEN_MATH_PLUS = "qwen-math-plus"
    QWEN_MATH_TURBO = "qwen-math-turbo"
    QWEN_CODER_TURBO = "qwen-coder-turbo"
    QWEN_2_5_CODER_32B = "qwen2.5-coder-32b-instruct"
    QWEN_2_5_VL_72B = "qwen2.5-vl-72b-instruct"
    QWEN_2_5_72B = "qwen2.5-72b-instruct"
    QWEN_2_5_32B = "qwen2.5-32b-instruct"
    QWEN_2_5_14B = "qwen2.5-14b-instruct"
    QWEN_QWQ_32B = "qwq-32b-preview"
    QWEN_QVQ_72B = "qvq-72b-preview"
    QWEN_QWQ_PLUS = "qwq-plus"

    # Yi models (01-ai)
    YI_LIGHTNING = "yi-lightning"
    YI_LARGE = "yi-large"
    YI_MEDIUM = "yi-medium"
    YI_LARGE_TURBO = "yi-large-turbo"
    YI_VISION = "yi-vision"
    YI_MEDIUM_200K = "yi-medium-200k"
    YI_SPARK = "yi-spark"
    YI_LARGE_RAG = "yi-large-rag"
    YI_LARGE_FC = "yi-large-fc"

    # DeepSeek models
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"
    # InternLM models
    INTERNLM3_LATEST = "internlm3-latest"
    INTERNLM3_8B_INSTRUCT = "internlm3-8b-instruct"
    INTERNLM2_5_LATEST = "internlm2.5-latest"
    INTERNLM2_PRO_CHAT = "internlm2-pro-chat"

    # Moonshot models
    MOONSHOT_V1_8K = "moonshot-v1-8k"
    MOONSHOT_V1_32K = "moonshot-v1-32k"
    MOONSHOT_V1_128K = "moonshot-v1-128k"

    # SiliconFlow models support tool calling
    SILICONFLOW_DEEPSEEK_V2_5 = "deepseek-ai/DeepSeek-V2.5"
    SILICONFLOW_DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
    SILICONFLOW_INTERN_LM2_5_20B_CHAT = "internlm/internlm2_5-20b-chat"
    SILICONFLOW_INTERN_LM2_5_7B_CHAT = "internlm/internlm2_5-7b-chat"
    SILICONFLOW_PRO_INTERN_LM2_5_7B_CHAT = "Pro/internlm/internlm2_5-7b-chat"
    SILICONFLOW_QWEN2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct"
    SILICONFLOW_QWEN2_5_32B_INSTRUCT = "Qwen/Qwen2.5-32B-Instruct"
    SILICONFLOW_QWEN2_5_14B_INSTRUCT = "Qwen/Qwen2.5-14B-Instruct"
    SILICONFLOW_QWEN2_5_7B_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
    SILICONFLOW_PRO_QWEN2_5_7B_INSTRUCT = "Pro/Qwen/Qwen2.5-7B-Instruct"
    SILICONFLOW_THUDM_GLM_4_9B_CHAT = "THUDM/glm-4-9b-chat"
    SILICONFLOW_PRO_THUDM_GLM_4_9B_CHAT = "Pro/THUDM/glm-4-9b-chat"

    # AIML models support tool calling
    AIML_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    AIML_MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.1"

    def __str__(self):
        return self.value

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            Model.GPT_3_5_TURBO,
            Model.GPT_4,
            Model.GPT_4_TURBO,
            Model.GPT_4O,
            Model.GPT_4O_MINI,
            Model.O1,
            Model.O1_PREVIEW,
            Model.O1_MINI,
            Model.O3_MINI,
            Model.GPT_4_5_PREVIEW,
        }

    @property
    def is_azure_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model
        from Azure.
        """
        return self in {
            Model.GPT_3_5_TURBO,
            Model.GPT_4,
            Model.GPT_4_TURBO,
            Model.GPT_4O,
            Model.GPT_4O_MINI,
        }

    @property
    def is_zhipuai(self) -> bool:
        r"""Returns whether this type of models is an ZhipuAI model."""
        return self in {
            Model.GLM_3_TURBO,
            Model.GLM_4,
            Model.GLM_4V,
            Model.GLM_4V_FLASH,
            Model.GLM_4V_PLUS_0111,
            Model.GLM_4_PLUS,
            Model.GLM_4_AIR,
            Model.GLM_4_AIR_0111,
            Model.GLM_4_AIRX,
            Model.GLM_4_LONG,
            Model.GLM_4_FLASHX,
            Model.GLM_4_FLASH,
            Model.GLM_ZERO_PREVIEW,
        }

    @property
    def is_anthropic(self) -> bool:
        r"""Returns whether this type of models is Anthropic-released model.

        Returns:
            bool: Whether this type of models is anthropic.
        """
        return self in {
            Model.CLAUDE_INSTANT_1_2,
            Model.CLAUDE_2_0,
            Model.CLAUDE_2_1,
            Model.CLAUDE_3_OPUS,
            Model.CLAUDE_3_SONNET,
            Model.CLAUDE_3_HAIKU,
            Model.CLAUDE_3_5_SONNET,
            Model.CLAUDE_3_5_HAIKU,
            Model.CLAUDE_3_7_SONNET,
        }

    @property
    def is_groq(self) -> bool:
        r"""Returns whether this type of models is served by Groq."""
        return self in {
            Model.GROQ_LLAMA_3_1_8B,
            Model.GROQ_LLAMA_3_3_70B,
            Model.GROQ_LLAMA_3_3_70B_PREVIEW,
            Model.GROQ_LLAMA_3_8B,
            Model.GROQ_LLAMA_3_70B,
            Model.GROQ_MIXTRAL_8_7B,
            Model.GROQ_GEMMA_2_9B_IT,
        }

    @property
    def is_together(self) -> bool:
        r"""Returns whether this type of models is served by Together AI."""
        return self in {
            Model.TOGETHER_LLAMA_3_1_405B,
            Model.TOGETHER_LLAMA_3_1_70B,
            Model.TOGETHER_LLAMA_3_3_70B,
            Model.TOGETHER_LLAMA_3_3_70B,
            Model.TOGETHER_MISTRAL_7B,
            Model.TOGETHER_MIXTRAL_8_7B,
        }

    @property
    def is_sambanova(self) -> bool:
        r"""Returns whether this type of model is served by SambaNova AI."""
        return self in {
            Model.SAMBA_LLAMA_3_1_8B,
            Model.SAMBA_LLAMA_3_1_70B,
            Model.SAMBA_LLAMA_3_1_405B,
        }

    @property
    def is_mistral(self) -> bool:
        r"""Returns whether this type of models is served by Mistral."""
        return self in {
            Model.MISTRAL_LARGE,
            Model.MISTRAL_NEMO,
            Model.MISTRAL_CODESTRAL,
            Model.MISTRAL_7B,
            Model.MISTRAL_MIXTRAL_8x7B,
            Model.MISTRAL_MIXTRAL_8x22B,
            Model.MISTRAL_CODESTRAL_MAMBA,
            Model.MISTRAL_PIXTRAL_12B,
            Model.MISTRAL_8B,
            Model.MISTRAL_3B,
        }

    @property
    def is_nvidia(self) -> bool:
        r"""Returns whether this type of models is a NVIDIA model."""
        return self in {
            Model.NVIDIA_NEMOTRON_340B_INSTRUCT,
            Model.NVIDIA_NEMOTRON_340B_REWARD,
            Model.NVIDIA_YI_LARGE,
            Model.NVIDIA_MISTRAL_LARGE,
            Model.NVIDIA_LLAMA3_70B,
            Model.NVIDIA_MIXTRAL_8X7B,
            Model.NVIDIA_LLAMA3_1_8B_INSTRUCT,
            Model.NVIDIA_LLAMA3_1_70B_INSTRUCT,
            Model.NVIDIA_LLAMA3_1_405B_INSTRUCT,
            Model.NVIDIA_LLAMA3_2_1B_INSTRUCT,
            Model.NVIDIA_LLAMA3_2_3B_INSTRUCT,
            Model.NVIDIA_LLAMA3_3_70B_INSTRUCT,
        }

    @property
    def is_gemini(self) -> bool:
        r"""Returns whether this type of models is Gemini model.

        Returns:
            bool: Whether this type of models is gemini.
        """
        return self in {
            Model.GEMINI_2_0_FLASH,
            Model.GEMINI_1_5_FLASH,
            Model.GEMINI_1_5_PRO,
            Model.GEMINI_2_0_FLASH_THINKING,
            Model.GEMINI_2_0_PRO_EXP,
            Model.GEMINI_2_0_FLASH_LITE_PREVIEW,
        }

    @property
    def is_reka(self) -> bool:
        r"""Returns whether this type of models is Reka model.

        Returns:
            bool: Whether this type of models is Reka.
        """
        return self in {
            Model.REKA_CORE,
            Model.REKA_EDGE,
            Model.REKA_FLASH,
        }

    @property
    def is_cohere(self) -> bool:
        r"""Returns whether this type of models is a Cohere model.

        Returns:
            bool: Whether this type of models is Cohere.
        """
        return self in {
            Model.COHERE_COMMAND_R_PLUS,
            Model.COHERE_COMMAND_R,
            Model.COHERE_COMMAND_LIGHT,
            Model.COHERE_COMMAND,
            Model.COHERE_COMMAND_NIGHTLY,
        }

    @property
    def is_yi(self) -> bool:
        r"""Returns whether this type of models is Yi model.

        Returns:
            bool: Whether this type of models is Yi.
        """
        return self in {
            Model.YI_LIGHTNING,
            Model.YI_LARGE,
            Model.YI_MEDIUM,
            Model.YI_LARGE_TURBO,
            Model.YI_VISION,
            Model.YI_MEDIUM_200K,
            Model.YI_SPARK,
            Model.YI_LARGE_RAG,
            Model.YI_LARGE_FC,
        }

    @property
    def is_qwen(self) -> bool:
        return self in {
            Model.QWEN_MAX,
            Model.QWEN_PLUS,
            Model.QWEN_TURBO,
            Model.QWEN_LONG,
            Model.QWEN_VL_MAX,
            Model.QWEN_VL_PLUS,
            Model.QWEN_MATH_PLUS,
            Model.QWEN_MATH_TURBO,
            Model.QWEN_CODER_TURBO,
            Model.QWEN_2_5_CODER_32B,
            Model.QWEN_2_5_VL_72B,
            Model.QWEN_2_5_72B,
            Model.QWEN_2_5_32B,
            Model.QWEN_2_5_14B,
            Model.QWEN_QWQ_32B,
            Model.QWEN_QVQ_72B,
            Model.QWEN_QWQ_PLUS,
        }

    @property
    def is_deepseek(self) -> bool:
        return self in {
            Model.DEEPSEEK_CHAT,
            Model.DEEPSEEK_REASONER,
        }

    @property
    def is_internlm(self) -> bool:
        return self in {
            Model.INTERNLM3_LATEST,
            Model.INTERNLM3_8B_INSTRUCT,
            Model.INTERNLM2_5_LATEST,
            Model.INTERNLM2_PRO_CHAT,
        }

    @property
    def is_moonshot(self) -> bool:
        return self in {
            Model.MOONSHOT_V1_8K,
            Model.MOONSHOT_V1_32K,
            Model.MOONSHOT_V1_128K,
        }

    @property
    def is_sglang(self) -> bool:
        return self in {
            Model.SGLANG_LLAMA_3_1_8B,
            Model.SGLANG_LLAMA_3_1_70B,
            Model.SGLANG_LLAMA_3_1_405B,
            Model.SGLANG_LLAMA_3_2_1B,
            Model.SGLANG_MIXTRAL_NEMO,
            Model.SGLANG_MISTRAL_7B,
            Model.SGLANG_QWEN_2_5_7B,
            Model.SGLANG_QWEN_2_5_32B,
            Model.SGLANG_QWEN_2_5_72B,
        }

    @property
    def is_siliconflow(self) -> bool:
        return self in {
            Model.SILICONFLOW_DEEPSEEK_V2_5,
            Model.SILICONFLOW_DEEPSEEK_V3,
            Model.SILICONFLOW_INTERN_LM2_5_20B_CHAT,
            Model.SILICONFLOW_INTERN_LM2_5_7B_CHAT,
            Model.SILICONFLOW_PRO_INTERN_LM2_5_7B_CHAT,
            Model.SILICONFLOW_QWEN2_5_72B_INSTRUCT,
            Model.SILICONFLOW_QWEN2_5_32B_INSTRUCT,
            Model.SILICONFLOW_QWEN2_5_14B_INSTRUCT,
            Model.SILICONFLOW_QWEN2_5_7B_INSTRUCT,
            Model.SILICONFLOW_PRO_QWEN2_5_7B_INSTRUCT,
            Model.SILICONFLOW_THUDM_GLM_4_9B_CHAT,
            Model.SILICONFLOW_PRO_THUDM_GLM_4_9B_CHAT,
        }

    @property
    def is_aiml(self) -> bool:
        return self in {
            Model.AIML_MIXTRAL_8X7B,
            Model.AIML_MISTRAL_7B_INSTRUCT,
        }
