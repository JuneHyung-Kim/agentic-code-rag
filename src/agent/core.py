import json
from typing import List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable

from config import config
from tools.search_tool import SearchTool

class CodeAgent:
    def __init__(self):
        self.search_tool = SearchTool()
        self.provider = config.model_provider
        
        if self.provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=config.openai_api_key)
            self.model_name = config.model_name
            self.tools = [self.search_tool.get_tool_definition()]
            self.messages = [
                {"role": "system", "content": "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."}
            ]
        elif self.provider == "gemini":
            if not config.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            genai.configure(api_key=config.gemini_api_key)
            
            # Map the function directly for Gemini
            self.tools = [self.search_tool.search_codebase]
            
            system_instruction = "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."
            
            self.model = genai.GenerativeModel(
                model_name=config.model_name if config.model_name != "gpt-4o" else "gemini-1.5-flash",
                tools=self.tools,
                system_instruction=system_instruction
            )
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(self, user_input: str) -> str:
        if self.provider == "openai":
            return self._chat_openai(user_input)
        elif self.provider == "gemini":
            return self._chat_gemini(user_input)

    def _chat_openai(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            self.messages.append(response_message)

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "search_codebase":
                        tool_output = self.search_tool.search_codebase(**function_args)
                        
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_output,
                        })
            else:
                return response_message.content

    def _chat_gemini(self, user_input: str) -> str:
        try:
            response = self.chat_session.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"

    def reset(self):
        if self.provider == "openai":
            self.messages = [
                {"role": "system", "content": "You are an expert AI software engineer. You have access to a codebase and can search it to answer questions. Always verify your assumptions by searching the code. When answering, reference specific files and lines if possible."}
            ]
        elif self.provider == "gemini":
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)

