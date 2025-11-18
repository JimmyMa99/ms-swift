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


@dataclass
class UserLMTemplateMeta(TemplateMeta):

    template_type = 'userlm' 

    prefix: Prompt = field(default_factory=list)
    

    prompt: Prompt = field(default_factory=lambda: [
        '<|start_header_id|>user<|end_header_id|>\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n'
    ])
    
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])

    suffix: Prompt = field(default_factory=lambda: [
        '<|eot_id|>',
        '<|start_header_id|>user<|end_header_id|>'
    ])


    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: [
            '<|start_header_id|>system<|end_header_id|>\n{{SYSTEM}}<|eot_id|>'
        ]
    )
    
    default_system: Optional[str] = None


register_template(
    UserLMTemplateMeta(
        template_type=LLMTemplateType.userlm
    )
)