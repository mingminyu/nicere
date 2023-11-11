# coding: utf8
import os
import re
import time
import yaml
import asyncio
import logging
from rich.logging import RichHandler
from typing import List, Dict, Literal, Optional, Union, Callable, Any
from pydantic import BaseModel, computed_field
from functools import wraps
from rich.pretty import pprint

"""
基于 Python re 库的扩展库，增强正则识别能力
"""


logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)


def compute_cost_time(func: Callable):
    """计算运算时长的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        if asyncio.iscoroutinefunction(func):
            res = await func(*args, **kwargs)
        else:
            res = func(*args, **kwargs)

        cost_time = round((time.time() - start_time) * 1000, 6)
        text = f"Function [bold red]{func.__name__}[/] costs {cost_time} ms"
        RulesSearcher.log.info(text, extra={"markup": True})

        return res
    return wrapper


class RegexRule(BaseModel):
    reg: re.Pattern
    conf: float = 0.0
    text_len: int = 0


class GroupRegexRule(BaseModel):
    rule_id: str
    regexes: List[RegexRule]
    positive: bool = True


class SceneRule(BaseModel):
    intent: str
    scene_id: str
    risk_level: str
    formula: str
    scene_rules: List[GroupRegexRule]
    enable: bool = True
    role: Literal['user', 'agent', 'all'] = "agent"
    global_formula: str = ""
    global_rules: List[GroupRegexRule] = []
    description: str = ""

    @computed_field
    def risk_level_id(self) -> int:
        """风险等级对应的 ID"""
        if self.risk_level == "高":
            return 1
        elif self.risk_level == "中":
            return 2
        else:
            return 3

    @computed_field
    def instruction(self) -> str:
        """LLM Instruction Prompt"""
        return f"请判断给出的文本内容是否命中{self.description}，输出为 1 或 0，以下是具体内容-->"

    @computed_field
    def highlight_text_pattern(self) -> re.Pattern:
        """高亮文本 Pattern"""
        highlight_pattern = "|".join(
            [regex.reg.pattern for scene_rule in self.scene_rules if scene_rule.positive
             for regex in scene_rule.regexes]
        )
        return re.compile(highlight_pattern)


class SceneMatchResult(BaseModel):
    scene_id: Optional[str] = None
    is_matched: bool = False
    rule_match_details: Optional[Dict[str, bool]] = None
    global_mode: bool = False


class RulesSearcher:
    __slots__ = ('rules', )

    log = logging.getLogger("rich")

    def __init__(
            self,
            rules_yaml_filepath: str
    ):
        if not os.path.exists(rules_yaml_filepath):
            raise ValueError("`rules_yaml_filepath` must be available.")

        self.log.warning("If you use `on` or `off` as a key, it will not work properly.")

        with open(rules_yaml_filepath, 'r', encoding='utf8') as f:
            rules_content = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.log.info(f'RegexSearcher version: {rules_content["version"]}')

        if "scenes" not in rules_content:
            raise ValueError("The content of `rules_yaml_filepath` must include `scenes` key.")

        self.rules = [SceneRule(**rule) for rule in rules_content["scenes"]]

    def get_scene_role(self, scene_id: str) -> Union[SceneRule, None]:
        """根据 scene_id 获取对应的 scene_rule"""
        for scene_rule in self.rules:
            if scene_rule.scene_id == scene_id:
                return scene_rule

        self.log.warning(f"scene_id `{scene_id}` is not exists.")
        return None

    @staticmethod
    async def regex_match(regex: RegexRule, text: str):
        """正则匹配"""
        if len(text) < regex.text_len:
            return False

        match_text = regex.reg.search(text)
        if match_text is None:
            return False

        if len(match_text.group()) / len(text) < regex.conf:
            return False

        return True

    async def group_regex_match(self, group_regex: GroupRegexRule, text: str) -> bool:
        """正则组批量匹配"""
        for regex in group_regex.regexes:
            res = await self.regex_match(regex, text)

            if res is True:
                return True

        return False

    @compute_cost_time
    async def is_sentence_match(
            self,
            text: str = "",
            role: str = "agent",
            *,
            exclude_scene_ids: List[str] = None,
            show_rule_match_detail: bool = False,
            global_match: bool = False
    ) -> List[SceneMatchResult]:
        """是否单句匹配
        :param text: 传入的文本
        :param role: 应用的角色
        :param exclude_scene_ids:  不需要匹配匹配的 scene id
        :param show_rule_match_detail: 显示每个 rule_id 命中具体信息
        :param global_match: 是否全局匹配
        """
        if exclude_scene_ids is None:
            exclude_scene_ids = []

        scenes_matched: List[SceneMatchResult] = []
        filter_rules = [
            rule for rule in self.rules
            if rule.enable is True and role == rule.role and rule.scene_id not in exclude_scene_ids
        ]

        for rule in filter_rules:
            scene_match_result: SceneMatchResult = SceneMatchResult()

            if global_match is True:
                if rule.global_formula != "":
                    scene_rules_match_result = {
                        global_rule.rule_id: await self.group_regex_match(global_rule, text)
                        for global_rule in rule.global_rules
                    }
                    scene_is_matched = eval(
                        rule.global_formula.replace("!", "not ").format(**scene_rules_match_result))
                else:
                    scene_rules_match_result = {}
                    scene_is_matched = False

                scene_match_result.global_mode = True
            else:
                scene_rules_match_result = {
                    scene_rule.rule_id: await self.group_regex_match(scene_rule, text)
                    for scene_rule in rule.scene_rules
                }
                scene_is_matched = eval(
                    rule.formula.replace("!", "not ").format(**scene_rules_match_result))

            scene_match_result.is_matched = scene_is_matched
            scene_match_result.scene_id = rule.scene_id

            if show_rule_match_detail:
                scene_match_result.rule_match_details = scene_rules_match_result

            scenes_matched.append(scene_match_result)

        return scenes_matched

    @staticmethod
    def highlight_text(scene_rule: SceneRule, text: str) -> List[Dict[str, Union[str, int]]]:
        """高亮场景关键词"""
        if isinstance(scene_rule.highlight_text_pattern, re.Pattern):
            highlight_text_pattern: re.Pattern = scene_rule.highlight_text_pattern
            highlight_entities = [
                {"start": highlight_text.start(), "end": highlight_text.end(), "value": highlight_text.span()}
                for highlight_text in highlight_text_pattern.finditer(text)
            ]
            return highlight_entities
        return []

    def is_context_match(
            self,
    ):
        """是否上下文对话满足正则匹配
        :TODO:
        """
        ...

    def validate_singe_regex(self, regex: RegexRule) -> bool:
        """验证单条正则是否书写正确"""
        err_cnt = 0

        if "，" in regex.reg.pattern:
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中存在 `，` 中文逗号!", extra={"markup": True})

        if "||" in regex.reg.pattern:
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中存在 `||` 符号!", extra={"markup": True})

        if "|)" in regex.reg.pattern:
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中存在 `|)` 符号!", extra={"markup": True})

        if regex.reg.pattern.count("(") != regex.reg.pattern.count(")"):
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中 `(` 和 `)` 数量不对等!", extra={"markup": True})

        if regex.reg.pattern.count("[") != regex.reg.pattern.count("]"):
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中 `[` 和 `]` 数量不对等!", extra={"markup": True})

        if re.search(r"{\d\.\d+}", regex.reg.pattern):
            err_cnt += 1
            self.log.error(f"{err_cnt}: 正则中存在 `{0.2}` 无效的表达符号!", extra={"markup": True})

        if err_cnt > 0:
            self.log.error(f"正则中存在 [red bold]{err_cnt}[/] 个错误!", extra={"markup": True})
            self.log.info(regex.reg)
            return False
        else:
            return True

    def validate_group_regex_rule(self, group_regex_rule: GroupRegexRule) -> bool:
        """验证正则组合是否正确"""
        self.log.info(f"Validating ruleId: [blue bold]{group_regex_rule.rule_id}[/]", extra={"markup": True})
        validation_result = [
            self.validate_singe_regex(regex_rule)
            for regex_rule in group_regex_rule.regexes
        ]

        if all(validation_result):
            self.log.info(f"Validating ruleId: [blue bold]{group_regex_rule.rule_id}[/], [green bold]ok[/]",
                          extra={"markup": True})
            return True
        else:
            self.log.info(f"Validating ruleId: [blue bold]{group_regex_rule.rule_id}[/], [red bold]error[/]",
                          extra={"markup": True})
            return False

    def validate_scene_rule(self, scene_rule: SceneRule) -> bool:
        """验证全部正则是否书写正确"""
        self.log.info(f"Validating sceneId: [blue bold]{scene_rule.scene_id}[/]", extra={"markup": True})

        validation_result = [
            self.validate_group_regex_rule(scene_rule_)
            for scene_rule_ in scene_rule.scene_rules
        ]

        if all(validation_result):
            self.log.info(f"Validating sceneId: [blue bold]{scene_rule.scene_id}[/], [green bold]ok[/]",
                          extra={"markup": True})
            return True
        else:
            self.log.info(
                (f"Validating sceneId: [blue bold]{scene_rule.scene_id}[/][blue bold]{scene_rule.scene_id}[/], "
                 f"[red bold]error[/]"),
                extra={"markup": True})
            return False

    def validate_scene_rules(self, scene_ids: Union[List[str], str] = None) -> None:
        """验证所有或指定场景的规则"""
        if isinstance(scene_ids, str):
            scene_ids = [scene_ids]
        elif scene_ids is None:
            scene_ids = [scene_rule.scene_id for scene_rule in self.rules]

        for scene_rule in self.rules:
            if scene_rule.scene_id in scene_ids:
                self.validate_scene_rule(scene_rule)
