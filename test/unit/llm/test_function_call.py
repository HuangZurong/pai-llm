import json
import os
import unittest

import openai
from loguru import logger

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL")
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"],
            }
        }
    }
]


class TestFunctionCall(unittest.TestCase):

    def test_function_call(self):
        messages = [
            {
                "role": "user",
                "content": "杭州今天天气怎么样？"
            }
        ]
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME"),
            messages=messages,
            tools=tools
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            t = tool_call.type
            call_id = tool_call.id
            function = tool_call.function
            function_name = function.name
            function_args = function.arguments

            if t != "function":
                continue

            if function_name == "get_weather":
                function_args = eval(function_args)
                location = function_args["location"]
                unit = function_args["unit"]
                function_response = {
                    "location": location,
                    "temperature": 22,
                    "unit": unit,
                    "description": "阴天"
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(function_response)
                })

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME"),
            messages=messages
        )
        logger.info(response.choices[0].message.content)
