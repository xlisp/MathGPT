"""
改进版 GSM8K 奖励 —— 替换 tasks/gsm8k.py 中的 extract_answer / reward。

改了三件事，对应面试里讲的"奖励层"：
  1. 鲁棒答案抽取：原版只认 "#### 数字"，模型推理对但格式没对就判 0 分，
     白白浪费正确 rollout。这里加 boxed{}、"answer is X"、末尾数字 三级回退。
  2. shaped reward：在 0/1 正确性之外，给"格式合法 / 调用了计算器工具"少量
     正奖励，缓解稀疏信号；给超长输出极小惩罚，抑制啰嗦/退化。
  3. 数值等价比较：把 "10"、"10.0"、"10.00" 视为相等，减少假错。

设计原则：correctness 仍是绝对主项（权重远大于 shaping），shaping 只在
正确性相同的 rollout 之间充当 tie-breaker，避免 reward hacking。
"""

import re

# 1) 三级答案抽取 ----------------------------------------------------------------
_GSM_RE   = re.compile(r"####\s*(-?[0-9\.\,]+)")
_BOXED_RE = re.compile(r"\\boxed\{\s*(-?[0-9\.\,]+)\s*\}")
_PHRASE_RE = re.compile(r"(?:answer|result|总共|等于|是)\D{0,8}(-?[0-9][0-9\.\,]*)", re.IGNORECASE)
_NUM_RE   = re.compile(r"-?[0-9][0-9\.\,]*")


def _normalize(num_str):
    if num_str is None:
        return None
    s = num_str.strip().replace(",", "").rstrip(".")
    if s in ("", "-", "."):
        return None
    try:
        # 数值归一：10 / 10.0 / 10.00 -> 同一个 float
        return float(s)
    except ValueError:
        return None


def extract_answer(completion):
    """按优先级回退抽取最终数值答案，返回 float 或 None。"""
    for rgx in (_GSM_RE, _BOXED_RE, _PHRASE_RE):
        m = rgx.search(completion)
        if m:
            val = _normalize(m.group(1))
            if val is not None:
                return val
    # 最后兜底：取正文里最后一个数字（常见于"...= 10"结尾）
    nums = _NUM_RE.findall(completion)
    for tok in reversed(nums):
        val = _normalize(tok)
        if val is not None:
            return val
    return None


# 2) 奖励整形 --------------------------------------------------------------------
_TOOL_RE = re.compile(r"<<[^>]+>>|<\|python_start\|>")  # 计算器工具调用痕迹


def shaped_reward(ref_text, response, *, max_tokens=256, n_tokens=None,
                  w_correct=1.0, w_format=0.05, w_tool=0.05, w_len=0.05):
    """
    返回 (total_reward, info_dict)。

    total = w_correct * correct
          + w_format  * has_clean_final_answer
          + w_tool     * used_calculator
          - w_len      * length_overflow_ratio
    其中 correct ∈ {0,1} 仍是主导项；其余三项最大合计 0.1，只在
    "对错相同"时区分质量，不会盖过正确性。
    """
    ref_num  = extract_answer(ref_text)
    pred_num = extract_answer(response)
    correct  = float(ref_num is not None and pred_num is not None and ref_num == pred_num)

    has_format = 1.0 if ("####" in response or "\\boxed" in response) else 0.0
    used_tool  = 1.0 if _TOOL_RE.search(response) else 0.0

    len_overflow = 0.0
    if n_tokens is not None and n_tokens > max_tokens:
        len_overflow = min(1.0, (n_tokens - max_tokens) / max(max_tokens, 1))

    total = (w_correct * correct
             + w_format * has_format
             + w_tool   * used_tool
             - w_len    * len_overflow)

    return total, {
        "correct": correct,
        "has_format": has_format,
        "used_tool": used_tool,
        "len_overflow": len_overflow,
    }
