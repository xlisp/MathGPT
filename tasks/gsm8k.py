"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
import os
from datasets import load_dataset, load_from_disk
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

# --- Agile 改进：鲁棒三级回退抽取（boxed / "answer is" / 末尾数字）---------------
# 设 MATHGPT_ROBUST_EXTRACT=0 可关闭，回退到官方仅认 "#### 数字" 的行为。
_BOXED_RE  = re.compile(r"\\boxed\{\s*(-?[0-9\.\,]+)\s*\}")
_PHRASE_RE = re.compile(r"(?:answer|result|总共|等于|是)\D{0,8}(-?[0-9][0-9\.\,]*)", re.IGNORECASE)
_NUM_RE    = re.compile(r"-?[0-9][0-9\.\,]*")

def _norm(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").rstrip(".")
    return s if s not in ("", "-", ".") else None

def extract_answer(completion):
    """
    Extract the numerical answer. 默认鲁棒模式按优先级回退：
      #### 数字  ->  \\boxed{数字}  ->  "answer is 数字"  ->  正文最后一个数字
    返回归一化后的字符串（与原版接口一致）。
    """
    match = GSM_RE.search(completion)
    if match:
        return _norm(match.group(1))
    if os.environ.get("MATHGPT_ROBUST_EXTRACT", "1") == "0":
        return None
    for rgx in (_BOXED_RE, _PHRASE_RE):
        m = rgx.search(completion)
        if m and _norm(m.group(1)) is not None:
            return _norm(m.group(1))
    nums = _NUM_RE.findall(completion)
    for tok in reversed(nums):
        if _norm(tok) is not None:
            return _norm(tok)
    return None

def _num_equal(a, b):
    """数值等价：10 / 10.0 / 10.00 视为相等。"""
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


class GSM8K(Task):

    def __init__(self, subset, split, offline_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        if offline_dir:
            self.ds = load_from_disk(os.path.join(offline_dir, "gsm8k", split)).shuffle(seed=42)
        else:
            self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
              We can handle this later possibly. For now just assume string.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int (数值等价比较，避免 10 vs 10.0 假错)
        is_correct = int(_num_equal(pred_num, ref_num))
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. 默认仍返回 0/1 正确性（与原版一致）。
        设 MATHGPT_SHAPED_REWARD=1 开启奖励整形：在正确性之外，对
        "格式合法 / 调用计算器" 给少量正奖励，缓解 0/1 稀疏信号。
        shaping 总量 ≤0.1，绝不盖过正确性主项，避免 reward hacking。
        """
        is_correct = float(self.evaluate(conversation, assistant_response))
        if os.environ.get("MATHGPT_SHAPED_REWARD", "0") != "1":
            return is_correct
        bonus = 0.0
        if "####" in assistant_response or "\\boxed" in assistant_response:
            bonus += 0.05  # 产出了清晰的最终答案
        if re.search(r"<<[^>]+>>|<\|python_start\|>", assistant_response):
            bonus += 0.05  # 使用了计算器工具
        return is_correct + bonus
