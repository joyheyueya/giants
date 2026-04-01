"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from verl.utils.reward_score.curriculum_math.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import re
THOUGHT_DELIMITER_END = "</think>"

def format_reward(solution_str, **kwargs):
    pattern = re.compile(r"""
        ^           # Start of the string.
        .*?         # Optional leading content (non-greedy).
        <notes>     # Opening <notes> tag.
        .*?         # Any content within notes (non-greedy, can be multi-line due to DOTALL).
        </notes>    # Closing </notes> tag.
        .*?         # Optional intermediate content between notes and think (non-greedy).
        <think>     # Opening <think> tag.
        .*?         # Any content within think (non-greedy, can be multi-line due to DOTALL).
        </think>    # Closing </think> tag.
        .*?         # Optional trailing content (non-greedy).
        $           # End of the string.
    """, re.DOTALL | re.VERBOSE)
    return 0.1 if pattern.match(solution_str) else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info):
        format_reward_score = format_reward(solution_str)
        if format_reward_score == 0.0:
            return 0. # format is incorrect
        
        model_answer = extract_answer(solution_str)
        if model_answer is None:
            return 0. # no answer found

        # Process the ground truth(s)
        ground_truths = ground_truth

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return format_reward_score

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return 1. + format_reward_score

        return format_reward_score



if __name__ == '__main__':
    solution_str = """<think>\n\nOkay, let's see. I need to find the value of the expression 2014² - 2013² + 2012² - 2011² + ... + 2² - 1². Hmm, this looks like a series of alternating squares. Maybe I can use a formula for the difference of squares here. Let me recall that a² - b² = (a - b)(a + b). So each term in the series can be written as (n+1)² - n², where n starts from 0 up to 2013. Wait, but the series alternates signs. So each term is positive when n is even and negative when n is odd. Let me write that out:\n\n2014² - 2013² + 2012² - 2011² + ... + 2² - 1².\n\nIf I group the terms in pairs, each pair would be (n+1)² - n². Let's see:\n\nFirst pair: 2014² - 2013² = (2014 - 2013)(2014 + 2013) = 1 * 4027 = 4027.\n\nSecond pair: 2012² - 2011² = (2012 - 2011)(2012 + 2011) = 1 * 4023 = 4023.\n\nThird pair: 2010² - 2009² = 1 * 4019 = 4019.\n\nAnd so on, until the last pair which is 2² - 1² = 1 * 3 = 3.\n\nSo the entire expression is the sum of these pairs: 4027 + 4023 + 4019 + ... + 3.\n\nBut wait, how many pairs are there? Since we start at 2014 and go down to 2, each pair is two terms. So the number of pairs is (2014 - 2)/2 + 1 = 2012/2 + 1 = 1006 + 1 = 1007 pairs. Because the first term is 2014, which is the first pair, then 2012, 2010, etc., down to 2. So each pair is two terms, so 2014 divided by 2 is 1007, but since we start at 2014 and go down to 2, we have 1007 pairs.\n\nSo the sum is 1007 terms, each of which is 4027, 4023, 4019, ..., 3. This is an arithmetic series where the first term a = 4027, the last term l = 3, and the number of terms n = 1007. The sum of an arithmetic series is given by S = n/2 * (a + l). Let me check that:\n\nSum = 1007/2 * (4027 + 3) = 1007/2 * 4030 = 1007 * 2015. Let me compute that. 1007 * 2015. Hmm, 1000 * 2015 = 2,015,000, and 7 * 2015 = 14,105. So total is 2,015,000 + 14,105 = 2,029,105. So the sum is 2,029,105.\n\nWait, but let me verify this step by step to make sure I didn't make a mistake.\n\nFirst, each pair is (n+1)² - n² = (n+1 - n)(n+1 + n) = 1 * (2n + 1) = 2n + 1. So the first pair is 2014² - 2013² = 2*2013 + 1 = 4027. The second pair is 2012² - 2011² = 2*2011 + 1 = 4023. The third pair is 2010² - 2009² = 2*2009 + 1 = 4019. And so on. So each pair is 2n + 1, where n starts at 2013 and decreases by 1 each time. So the first term is 2*2013 + 1 = 4027, the second term is 2*2011 + 1 = 4023, etc., down to 2*1 + 1 = 3. So the sequence is 4027, 4023, 4019, ..., 3. This is an arithmetic sequence with first term 4027, last term 3, and common difference -4. Let me check the number of terms. The number of terms is (4027 - 3)/4 + 1 = 4024/4 + 1 = 1006 + 1 = 1007. So the number of terms is 1007. The sum is 1007/2 * (4027 + 3) = 1007 * 2015. Let me compute 1007 * 2015. 1000 * 2015 = 2,015,000, 7 * 2015 = 14,105. So total is 2,029,105. So that seems correct.\n\nAlternatively, maybe I can compute the sum directly by pairing terms. Let's see:\n\nFirst pair: 2014² - 2013² = 4027.\n\nSecond pair: 2012² - 2011² = 4023.\n\nThird pair: 2010² - 2009² = 4019.\n\n...\n\n2014th term: 2² - 1² = 3.\n\nSo the sum is 4027 + 4023 + 4019 + ... + 3. Let's see, this is an arithmetic series with first term 4027, last term 3, and common difference -4. The number of terms is 1007. The sum is 1007/2 * (4027 + 3) = 1007 * 2015 = 2,029,105. So that's the same result.\n\nTherefore, the value of the expression is 2,029,105.\n\n</think>\n\n<answer>\n\nThe given expression is \\(2014^2 - 2013^2 + 2012^2 - 2011^2 + \\ldots + 2^2 - 1^2\\). This can be simplified using the difference of squares formula:  \n\\[\na^2 - b^2 = (a - b)(a + b).\n\\]  \nEach pair \\((n+1)^2 - n^2\\) simplifies to \\(2n + 1\\). The sequence of terms is \\(4027, 4023, 4019, \\ldots, 3\\), which forms an arithmetic series with:  \n- First term (\\(a\\)) = 4027  \n- Last term (\\(l\\)) = 3  \n- Common difference (\\(d\\)) = -4  \n- Number of terms (\\(n\\)) = 1007  \n\nThe sum of an arithmetic series is given by:  \n\\[\nS = \\frac{n}{2} \\times (a + l).\n\\]  \nSubstituting the values:  \n\\[\nS = \\frac{1007}{2} \\times (4027 + 3) = \\frac{1007}{2} \\times 4030 = 1007 \\times 2015 = 2,029,105.\n\\]  \n\n**Final Answer:**  \n\\boxed{2029105}\n\n</answer>""" 
    score = compute_score(
        data_source='math',
        solution_str=solution_str,
        ground_truth="2029105",
        extra_info=None
    )
    print(score)
