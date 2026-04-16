# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"


EVAL_AGENT_USER_PROMPT = """
Now, here is a smartphone operation task description:
**{task_description}**
{history_info}
Please carefully determine whether the task has been correctly and completely executed according to the provided screenshots. Use 1 to indicate success and 0 to indicate failure.

{text_on_screenshot}

Use the following JSON format for your response:
{{
    "reason": "<Brief description of why you believe the task was successful or failed, including the alignment or misalignment between the task description and screenshots, starting with I believe this task is successful/failed>",
    "result": "<1 OR 0>"
}}

Remember:
- Do not make assumptions based on information not presented in the screenshots. Only evaluate what is explicitly shown.
- Ensure that every entity and action in the task description is precisely matched and fulfilled.
- Consider additional actions taken after a task is successfully completed as part of the success, as long as those actions don’t impact the task's completion or cause failure.
- A filtering subtask is only correct when a specific filter is applied as a feature of the app. Using the criteria as a keyword search will cause the subtask to fail.
- Subtasks can be completed in any order unless they are explicitly dependent on each other.
- Subtasks completed correctly mid-process, even if not reflected in the final screenshot, should be considered successful.
- Subtasks that initially appear to fail but are corrected by subsequent actions should be considered successful.
- A task can be considered successful even if some subtasks are not completed in one go, as long as the final result meets the task requirements.
- Focus on the overall objective of the task without being distracted by minor, irrelevant details.
- Pay attention to subtle UI differences that might indicate task completion or failure, such as highlighted tabs or changes in font.
{text_on_screenshot_attn}
- Return exactly one JSON object. No extra text.
- The JSON **must** be syntactically valid and parseable.
""".strip()


INCLUDE_TEXT_INFO = """
To assist you in determining whether the task was successful, action information is provided. Use this information only when you cannot determine success purely based on the screenshots. The action information on the i-th screenshot describes the changes from the i-th screenshot to the i+1-th screenshot, while the last screenshot contains no action information as the task ends afterward. This information is presented as a white strip attached to the original screenshot, separated by a blue line. In some screenshots, a red dot may indicate where a specific action occurred (e.g., clicked or long-pressed), triggering an event or interaction.
""".strip()


INCLUDE_TEXT_ATTN = """
- Consider the action information only when necessary.
- Pop-ups that appear immediately after an action may not be captured in the screenshots; do not consider this a failure.
- Some subtasks can be completed with a single action, such as clicking an icon that shuffles a playlist.
""".strip()


NOT_INCLUDE_TEXT_INFO = """
To assist you in determining whether the task was successful, action information is provided. Use this information only when you cannot determine success purely based on the screenshots. The i-th screenshot may contain details that change the screenshot from the i-th to the i+1-th, while the last screenshot contains no action information as the task ends afterward. In some screenshots, a red dot may indicate where a specific action occurred (e.g., clicked or long-pressed), triggering an event or interaction. If there isn't a red dot, the action is more complex than a single position operation (e.g., a swipe or text input). You can find the details of these actions below, if applicable.
{extra_action}
""".strip()


NOT_INCLUDE_TEXT_ATTN = """
- Consider the action information only when necessary.
- Pop-ups that appear immediately after an action may not be captured in the screenshots; do not consider this a failure.
- Some subtasks can be completed with a single action, such as clicking an icon that shuffles a playlist.
""".strip()


def get_action_mode_prompt(include_text: str,
                           extra_action: str = "") -> tuple[str, str]:
    if include_text == "true":
        return INCLUDE_TEXT_INFO, INCLUDE_TEXT_ATTN
    elif include_text == "false":
        return NOT_INCLUDE_TEXT_INFO.format(extra_action=extra_action), NOT_INCLUDE_TEXT_ATTN


APP_DECOMPOSITION_USER_PROMPT = """
Here is the task description:
```
{task_description}
```

Here is the app list:
```
{task_app}
```

Ensure the order of apps in your final output is exactly the same as the order provided in my app list.
""".strip()


SPLIT_DATA_USER_PROMPT = """
Now, for any smartphone control instruction, decompose the task into subtasks using the format above.

Task:
```
{task_description}
```

APP list:
```
{task_app}
```
""".strip()


MEMORY_SUMMARY_USER_PROMPT = """
Here is the description: 
```
{task_description}
```
""".strip()


SUMMARY_IMG_AGENT_USER_PROMPT = """
Batch: {batch_id} 

Task goal:
{task_description}

Image notes:
- These images belong to the same task flow (a screenshot sequence) or are candidate screens for the same task.
- You must:
  1) Produce one observation entry for EVERY image in this batch.
  2) Infer within-batch events (changes between images) only when supported by evidence.
- The unique key for each image is: b1_i1, b1_i2, ... in the exact order received.

Output rules:
- Output ONLY a single JSON object, strictly following the given schema.
- "observations" MUST include exactly one entry per image in this batch.
- "events" should include only changes you can justify; do not invent events.
""".strip()
