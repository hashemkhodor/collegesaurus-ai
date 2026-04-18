"""Manual ReAct-style loop over Gemini function calling.

Why not `chats.create` with `automatic_function_calling`?
---------------------------------------------------------
The automatic path introspects your Python functions to produce tool
declarations; when introspection fails it silently drops the tool and the
model then reports "no tool available." We declare schemas explicitly in
tools.py and run the tool-call loop here so every step is inspectable.

Loop shape
----------
1. Append user message → generate_content.
2. If the model returned a plain text Part, return it.
3. Otherwise the candidate contains one or more `function_call` Parts.
   Execute each via TOOL_REGISTRY, wrap the result in a `function_response`
   Part, append a `user`-role Content with those responses, loop.
4. Cap at MAX_STEPS so a misbehaving model can't spin forever.
"""

from google import genai
from google.genai import types

import config
from tools import TOOL_DECLARATIONS, TOOL_REGISTRY, _GEMINI


SYSTEM_PROMPT = (
    "You are Collegesaurus, an assistant that helps Lebanese high-school "
    "students (and their parents) choose a university or find a scholarship.\n"
    "\n"
    "Ground rules:\n"
    "1. ALWAYS call one of the provided tools before answering a factual "
    "question. Do not invent university names, majors, tuition figures, "
    "deadlines, or contact details — look them up.\n"
    "2. When you cite a fact from a tool result, include the Source URL it "
    "gave you. Keep citations concise.\n"
    "3. If the tools return no useful matches, say so plainly. Do not guess.\n"
    "4. Stay in scope: Lebanese higher education and the scholarships our "
    "dataset covers. For off-topic questions, say the site focuses on "
    "Lebanese universities and external scholarships, and suggest what you "
    "can help with.\n"
    "5. Prefer concise, structured answers: short paragraphs, bullet lists "
    "for options, bold key numbers (deadlines, costs).\n"
    "6. Reply in the language the user wrote in (English or Arabic)."
)

MAX_STEPS = 6


def _tool_config():
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
    )


def _execute_call(fc):
    """Run one FunctionCall from the model and wrap the result as a Part."""
    fn = TOOL_REGISTRY.get(fc.name)
    args = dict(fc.args or {})
    if fn is None:
        result = "Error: unknown tool {}".format(fc.name)
    else:
        try:
            result = fn(**args)
        except Exception as exc:
            # Return the error to the model so it can recover / retry.
            result = "Error: {}".format(exc)
    return types.Part.from_function_response(
        name=fc.name, response={"result": result}
    )


class ChatSession:
    """Thin session holding conversation history between turns.

    Stateless across Streamlit reruns — we just keep the list of Content
    objects and replay them each call.
    """

    def __init__(self):
        self.history: list[types.Content] = []

    def reset(self):
        self.history = []

    def send(self, user_text: str) -> str:
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        )
        cfg = _tool_config()

        for _ in range(MAX_STEPS):
            resp = _GEMINI.models.generate_content(
                model=config.GEMINI_CHAT_MODEL,
                contents=self.history,
                config=cfg,
            )
            candidate = resp.candidates[0]
            self.history.append(candidate.content)

            fn_calls = [
                p.function_call
                for p in (candidate.content.parts or [])
                if p.function_call
            ]
            if not fn_calls:
                # Plain text answer — we're done.
                return resp.text or "(empty response)"

            # Execute each requested tool and return the results in a single
            # user-role Content before looping.
            self.history.append(
                types.Content(
                    role="user",
                    parts=[_execute_call(fc) for fc in fn_calls],
                )
            )

        return "(stopped after {} tool steps without a final answer)".format(
            MAX_STEPS
        )


def build_chat() -> ChatSession:
    """Public constructor so app.py doesn't import the class name directly."""
    return ChatSession()
