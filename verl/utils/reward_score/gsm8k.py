# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

# Upper bound on how much of the tail of a solution we regex-search for the answer.
#
# This exists purely as a performance guard: regex over a very long string is slow, and for math
# problems the final answer is at the end. The previous value was 300 *characters* with a comment
# claiming it approximated 300 tokens -- but one token averages ~4 characters of English, so 300
# characters is only ~75 tokens. Any model that emitted "#### <answer>" and then kept writing for
# more than ~75 tokens had its answer silently clipped out of the search window and scored 0
# despite being correct. That is a systematic false negative, and it gets worse as a model's style
# drifts during RL. 1200 characters is the ~300 tokens that was originally intended.
_SOLUTION_CLIP_CHARS = 1200


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Only search the tail of the string; see _SOLUTION_CLIP_CHARS above.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def _answers_match(answer, ground_truth):
    """Compares two answer strings, numerically when both parse as numbers.

    Exact string equality rejects answers that are numerically correct but differently formatted
    -- "72.0", "72.00" and "72." all fail against a ground truth of "72". Those are correct
    solutions being scored 0, which both understates measured accuracy and injects label noise
    into the RL signal. Falls back to a whitespace-stripped string compare when either side is
    not numeric, so non-numeric ground truths behave exactly as before.
    """
    if answer is None:
        return False
    normalized = str(answer).replace(",", "").replace("$", "").strip()
    truth = str(ground_truth).replace(",", "").replace("$", "").strip()
    if normalized == truth:
        return True
    try:
        return abs(float(normalized) - float(truth)) < 1e-6
    except (ValueError, TypeError):
        return False


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if _answers_match(answer, ground_truth):
            return score
        else:
            return format_score

