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


# @dataclass
# class UserLMTemplateMeta(TemplateMeta):

#     template_type = 'userlm' 

#     prefix: Prompt = field(default_factory=list)
    

#     prompt: Prompt = field(default_factory=lambda: [
#         '<|start_header_id|>user<|end_header_id|>\n{{QUERY}}<|eot_id|>'
#         '<|start_header_id|>assistant<|end_header_id|>\n'
#     ])
    
#     chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])

#     suffix: Prompt = field(default_factory=lambda: [
#         '<|eot_id|>',
#         '<|start_header_id|>user<|end_header_id|>'
#     ])

#     response_prefix: str = '<|start_header_id|>user<|end_header_id|>\n'

#     system_prefix: Optional[Prompt] = field(
#         default_factory=lambda: [
#             '<|start_header_id|>system<|end_header_id|>\n{{SYSTEM}}<|eot_id|>'
#         ]
#     )
    
#     default_system: Optional[str] = None


# register_template(
#     UserLMTemplateMeta(
#         template_type=LLMTemplateType.userlm
#     )
# )
from ..template_inputs import StdTemplateInputs
from ..utils import Context, ContextType, StopWordsCriteria, fetch_one, findall, split_str_parts_by, Prompt

class UserLMIntentTemplate(Template):
    def _swift_encode(self, inputs: StdTemplateInputs) -> Tuple[List[Union[str, int]], List[float], int]:
        """
        UserLM 编码逻辑：
        1. 历史对话 (Context) -> Loss = 0.0
        2. 最后一条 User 回复 (Target, 仅训练时) -> Loss = 1.0
        3. 推理引导 -> 追加 User Header
        """
        messages = inputs.messages or []
        system = self._get_system(inputs)
        
        res_context_list: List[Union[str, int]] = []
        loss_scale_list: List[float] = [] # 手动管理 Loss Scale，避开基类检查
        res_context_types: List[ContextType] = [] 

        # --- 1. System 部分 (Loss=0) ---
        if system:
            system_str = f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
            res_context_list.append(system_str)
            loss_scale_list.append(0.0)
            res_context_types.append(ContextType.OTHER)

        # --- 2. 划分 Context 和 Target ---
        # 如果是训练模式，最后一条如果是 user，则它是 Label，其余是 Context
        # 如果是推理模式，全都是 Context
        
        context_messages = messages
        target_message = None
        
        if self.is_training and messages:
            # 检查最后一条是否是我们想学的 User 回复
            if messages[-1]['role'] == 'user':
                context_messages = messages[:-1]
                target_message = messages[-1]
            else:
                # 如果最后一条是 assistant，说明可能数据不对，或者不需要计算 loss
                # 这里视情况而定，通常 UserLM 训练数据最后一条必须是 User
                pass

        # --- 3. 构建历史上下文 (全 Input, Loss=0) ---
        for msg in context_messages:
            role = msg['role']
            content = msg['content']
            
            # 无论是 user 还是 assistant，只要在历史里，都是 context
            full_str = f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
            
            res_context_list.append(full_str)
            loss_scale_list.append(0.0)
            res_context_types.append(ContextType.OTHER)

        # --- 4. 构建训练目标 (Target, Loss=1) ---
        if target_message:
            role = target_message['role'] # 应该是 'user'
            content = target_message['content']
            
            # Header 不计算 Loss
            header = f"<|start_header_id|>{role}<|end_header_id|>\n"
            res_context_list.append(header)
            loss_scale_list.append(0.0)
            res_context_types.append(ContextType.OTHER)
            
            # Content 计算 Loss
            res_context_list.append(content)
            loss_scale_list.append(1.0)
            res_context_types.append(ContextType.RESPONSE)
            
            # EOT (停止符) 计算 Loss
            res_context_list.append("<|eot_id|>")
            loss_scale_list.append(1.0)
            res_context_types.append(ContextType.RESPONSE)

        # --- 5. 推理引导 (Suffix) ---
        # 如果不是在训练，且对话结束在 assistant，说明轮到 user 说话了
        if not self.is_training and context_messages:
            last_role = context_messages[-1]['role']
            if last_role == 'assistant':
                trigger = "<|start_header_id|>user<|end_header_id|>"
                res_context_list.append(trigger)
                loss_scale_list.append(0.0)
                res_context_types.append(ContextType.SUFFIX)

        # --- 6. 统计 Answer Length ---
        answer_len = 0
        if self.is_training:
            # 简单统计有多少个片段参与了 loss 计算 (Content + EOT)
            # 注意这只是片段数，不是 token 数，但通常足够用于日志
            answer_len = sum(1 for x in loss_scale_list if x > 0.0)

        return res_context_list, loss_scale_list, answer_len

@dataclass
class UserLMTemplateMeta(TemplateMeta):
    template_type = 'userlm' 
    
    # 必须指定 template_cls
    template_cls: Type[Template] = UserLMIntentTemplate
    
    # 保持默认值即可，核心逻辑都在 _swift_encode 里
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=list)
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])
    suffix: Prompt = field(default_factory=list)
    system_prefix: Optional[Prompt] = field(default_factory=list)

# 注册模板
register_template(
    # 直接实例化我们定义好的 Meta 类
    UserLMTemplateMeta(
        template_type=LLMTemplateType.userlm,
        template_cls=UserLMIntentTemplate
        # 关键：不要传递 template_cls 参数！
        # 这样框架就会使用它默认的、能够理解以上所有字段的 Template 类。
    )
)