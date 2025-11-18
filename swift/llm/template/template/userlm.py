# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Type

import json
import torch
from torch import nn

from ..base import Template
from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, ContextType, StopWordsCriteria, fetch_one, findall, split_str_parts_by, Prompt

class UserLMIntentTemplate(Template):
    def _swift_encode(self, inputs: StdTemplateInputs) -> Tuple[List[Union[str, int]], List[float], int]:
        """
        此方法实现了 UserLM 的双模式 prompt 构建逻辑：
        1. 对话续写模式 (Chat Mode): 当历史以 user 结尾时，提示 assistant 回复。
        2. 意图总结模式 (Summarization Mode): 当历史以 assistant 结尾或在微调时，
           添加特殊的 user 触发器 token。
        """

        messages = inputs.messages or []
        system = inputs.system

        # --- A. 区分上下文和目标（用于微调） ---
        context_messages = messages
        target_summary_message = None
        if self.is_training:
            if not messages or messages[-1]['role'] != 'assistant':
                raise ValueError(
                    "For fine-tuning, the last message must have role 'assistant' "
                    "and contain the target intent summary.")
            context_messages = messages[:-1]
            target_summary_message = messages[-1]

        # --- B. 初始化返回列表 ---
        res_context_list: List[Union[str, int]] = []
        res_context_types: List[ContextType] = []

        # --- C. 处理 system prompt ---
        if system:
            system_part_str = f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
            res_context_list.append(system_part_str)
            res_context_types.append(ContextType.OTHER)

        # --- D. 构建对话历史上下文 [CONVERSATION] ---
        for message in context_messages:
            role = message['role']
            content = message['content']
            part_str = f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            res_context_list.append(part_str)
            res_context_types.append(ContextType.OTHER)

        # --- E. 【核心逻辑】根据模式构建最终的触发器 token ---
        # 微调时，我们总是在训练总结任务。
        # 推理时，如果最后一条消息是 assistant，我们也认为是总结任务。
        is_summarization_mode = self.is_training or (context_messages and context_messages[-1]['role'] == 'assistant')

        if is_summarization_mode:
            # **意图总结模式**
            final_trigger_token = '<|start_header_id|>user<|end_header_id|>'
            res_context_list.append(final_trigger_token)
            res_context_types.append(ContextType.OTHER)
        else:
            # **对话续写模式**
            # 此时 context_messages 的最后一条一定是 user
            final_trigger_token = '<|start_header_id|>assistant<|end_header_id|>\n'
            res_context_list.append(final_trigger_token)
            res_context_types.append(ContextType.OTHER)


        # --- F. 添加目标（仅在微调时，此时必为总结模式） ---
        answer_len = 0
        if self.is_training and target_summary_message:
            target_summary = target_summary_message['content']
            res_context_list.append(target_summary)
            res_context_types.append(ContextType.RESPONSE)
            
            suffix = self.template_meta.suffix
            res_context_list += suffix
            res_context_types += [ContextType.SUFFIX] * len(suffix)
            
            answer_len = 1 + len(suffix)

        # --- G. 调用父类的 loss_scale 函数生成最终的 loss mask ---
        res_context_list, loss_scale_list = self.loss_scale(
            res_context_list, res_context_types, inputs.messages, **inputs.extra_kwargs)
        
        return res_context_list, loss_scale_list, answer_len
        
@dataclass
class UserLMTemplateMeta(TemplateMeta):
    template_type = 'userlm' 

    # suffix 对于训练至关重要，它通常是 EOS token
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])

    # 因为所有逻辑都在 _swift_encode 中，其他字段可以保持简单
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['{{QUERY}}']) 
    chat_sep: Optional[Prompt] = field(default_factory=list)
    system_prefix: Optional[Prompt] = field(default_factory=list)
    
    # 【关键】将 template_cls 指向我们自己实现的、支持微调的类
    template_cls: Type[Template] = UserLMIntentTemplate

# 注册模板
register_template(
    # 直接实例化我们定义好的 Meta 类
    UserLMTemplateMeta(
        template_type=LLMTemplateType.userlm
        
        # 关键：不要传递 template_cls 参数！
        # 这样框架就会使用它默认的、能够理解以上所有字段的 Template 类。
    )
)