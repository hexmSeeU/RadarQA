import asyncio
import re
from typing import List
from difflib import SequenceMatcher

import json

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)
            

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards

class MCQReward(ORM):

    def parse_option_and_content(self, answer_text):
        """
        将字符串分为选项和内容，同时：
        - 如果仅输入选项（单个字母，例如 C），则返回选项。
        - 支持多种分隔符，如 ':', '.', ',', 空格, ')'。
        - 如果仅有内容（无选项），返回 (None, 内容)。
        - 如果有选项和内容，返回 (选项, 内容)。

        参数:
        answer_text (str): 输入字符串。

        返回:
        tuple 或选项: 
            - (选项, 内容) 如果有选项和内容。
            - 选项 如果只有选项。
            - (None, 内容) 如果只有内容。
        """
        # 使用正则表达式匹配选项和内容
        # 匹配 A, B, C, D 或 a, b, c, d 后跟空格、冒号、句号等分隔符
        match = re.match(r'^\s*([abcdABCD])(?!\w)\s*[):.,\s]?\s*(.*)$', answer_text)
        
        if match:
            option = match.group(1).strip()  # 捕获选项
            content = match.group(2).strip()  # 捕获内容
            
            # 情况1: 如果没有内容，仅仅有选项，直接返回选项
            if not content:
                return option, None

            # 情况2: 如果有选项和内容，返回 (选项, 内容)
            return option, content
        else:
            # 情况3: 无选项但有内容，返回 (None, 内容)
            return None, answer_text.strip()
            
    def fuzzy_match(self, answer1, answer2, threshold=0.8):
        """
        判断两个字符串是否模糊匹配。
        """
        answer1 = answer1.lower()
        answer2 = answer2.lower()

        similarity = SequenceMatcher(None, answer1, answer2).ratio()

        return similarity>threshold

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            # try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    # sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    gt_string = sol
                    gt_option, gt_content = self.parse_option_and_content(gt_string)

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    st_option, st_content = self.parse_option_and_content(student_answer)

                    print("gt_pair:", gt_option, gt_content)
                    print("st_pair:", st_option, st_content)

                    if gt_option and st_option:
                        if st_option == gt_option:
                            print('good_answer', st_option, gt_option)
                            reward = 1.0
                        else:
                            print('bad_answer', gt_string, student_answer)
                    elif gt_content and st_content:
                        if st_content == gt_content:
                            print('good_answer', st_content, gt_content)
                            reward = 1.0
                        else:
                            print('bad_answer', gt_string, student_answer)
                    # Compare the extracted answers
                    else:
                        print('bad_answer', gt_string, student_answer)
                except Exception as e:
                    print('exception details:', e)
                    print('exception', content, sol)
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards

class WeatherRadarIQAReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                label_matches = re.findall(r'<(.*?)>', ground_truth)
                
                

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                pred_matches = re.findall(r'<(.*?)>', student_answer)
                if len(pred_matches) == len(label_matches):
                    hits = [i for i in range(len(pred_matches)) if pred_matches[i] == label_matches[i]]
                    reward = len(hits) / len(pred_matches)
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards
    
    
    
class WeatherRadarIQAReward_v2(ORM):      
    def extract_json_from_response(self, response):
        # 正则表达式提取 JSON 部分
        match = re.search(r'```json\n({.*})\n```', response, re.DOTALL)
        if match:
            return match.group(1)  # 返回匹配的 JSON 字符串
        else:
            return response  
    def valid_response_format(self, text):
        required_keys = [
            "Overall Performance",
            "Miss Performance",
            "False Alarm Performance",
            "Sharpness Performance",
            "High Value Performance"
        ]
        pattern = r'\'([^\']+)\'\s*:\s*\'([^\']+)\''
        matches = re.findall(pattern, text)
        
        # 创建字典并将所有键转为小写
        result_dict = {}
        for key, value in matches:
            result_dict[key] = value
        
        # 检查是否包含所有必需的键（忽略大小写）
        missing_keys = [key for key in required_keys if key not in result_dict]
        return len(missing_keys), result_dict
    
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try: 
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                cleaned_str = ground_truth.replace("'", "").replace("{", "").replace("}", "")
                # print(ground_truth)
                # 按逗号分割成键值对
                pairs = cleaned_str.split(", ")
                # 创建字典
                gt_result_dict = {}
                for pair in pairs:
                    key, value = pair.split(": ")
                    gt_result_dict[key] = value
                # print("gt_result_dict", gt_result_dict)
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                # print("stu_answers", student_answer)
                response_text = self.extract_json_from_response(student_answer)
                len_miss, result_dict = self.valid_response_format(response_text)
                
                if len_miss == 0:
                    required_keys = list(gt_result_dict.keys())
                    hit = 0
                    hits = [i for i in range(len(required_keys)) if result_dict[required_keys[i]] == gt_result_dict[required_keys[i]]]
                    reward = len(hits) / len(required_keys)
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


class jsonformat(ORM):
    def extract_json_from_response(self, response):
        # 正则表达式提取 JSON 部分
        match = re.search(r'```json\n({.*})\n```', response, re.DOTALL)
        if match:
            return match.group(1)  # 返回匹配的 JSON 字符串
        else:
            return response  # 如果没有匹配，直接返回原始字符串
    def valid_response_format(self, text, required_keys):
        # required_keys = [
        #     "Overall Performance",
        #     "Miss Performance",
        #     "False Alarm Performance",
        #     "Sharpness Performance",
        #     "High Value Performance"
        # ]
        pattern = r'\'([^\']+)\'\s*:\s*\'([^\']+)\''
        matches = re.findall(pattern, text)
        # print(text)
        # 创建字典并将所有键转为小写
        result_dict = {}
        for key, value in matches:
            result_dict[key] = value
        # print("result_dict", result_dict)
        # 检查是否包含所有必需的键（忽略大小写）
        missing_keys = [key for key in required_keys if key not in result_dict]
        return len(missing_keys)
    
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try: 
                # Extract answer from content if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                cleaned_str = ground_truth.replace("'", "").replace("{", "").replace("}", "")
                # print(ground_truth)
                # 按逗号分割成键值对
                pairs = cleaned_str.split(", ")
                # 创建字典
                gt_result_dict = {}
                for pair in pairs:
                    key, value = pair.split(": ")
                    gt_result_dict[key] = value
                
                required_keys = list(gt_result_dict.keys())
                
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                response_text = self.extract_json_from_response(student_answer)
                len_miss = self.valid_response_format(student_answer, required_keys)
                
                reward = 1 - len_miss / 5
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        # print("json reward", reward)
        return rewards
        
                
                
    





orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_science_mcq'] = MCQReward
orms['external_weatherradariqa'] = WeatherRadarIQAReward
orms['external_weatherradariqa_v2'] = WeatherRadarIQAReward_v2
orms['jsonformat'] = jsonformat
