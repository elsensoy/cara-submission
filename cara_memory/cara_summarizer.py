# cara_summarizer.py

import json
from datetime import datetime
from cara_memory import save_session_entry  # from previous file

# ---------- 1. Small helper for timestamp ----------

def current_iso_timestamp():
    return datetime.now().replace(microsecond=0).isoformat()

# ---------- 2. Your LLM client wrapper (placeholder) ----------

def call_llm_for_summary(system_prompt: str, instruction_text: str, conversation_log):
    """
    Replace this with your actual Gemini / LLM API call.

    system_prompt: the system-level instructions (Cara memory summarizer persona)
    instruction_text: the text that explains the JSON schema
    conversation_log: list of {"role": "user"/"assistant", "content": "..."}

    Should return the raw string of the model's reply (which must be JSON).
    """
    # PSEUDOCODE example using a "client" you would already have:
    #
    # response = client.generate(
    #     system=system_prompt,
    #     messages=[
    #         {"role": "user", "content": instruction_text},
    #         {"role": "user", "content": json.dumps({"conversation_log": conversation_log})}
    #     ]
    # )
    # return response.text
    #
    # For now we leave it abstract:
    raise NotImplementedError("Implement this with your Gemini or other LLM call.")


# ---------- 3. Main function: summarize + save ----------

def summarize_conversation_for_memory(conversation_log):
    """
    conversation_log: list[dict] like:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    Returns the path to the saved session JSON (via save_session_entry).
    """

    system_prompt = (
        "You are a memory-summarizing assistant for an AI teddy bear companion named Cara.\n"
        "Your task:\n"
        "- Read a full conversation between the user and Cara.\n"
        "- Extract the emotionally and personally important parts.\n"
        "- Output a single JSON object describing this conversation in a stable schema.\n\n"
        "Rules:\n"
        "- DO NOT include any explanation, comments, or extra text.\n"
        "- Your entire reply must be valid JSON, matching the schema exactly.\n"
        "- Do not invent memories that are not supported by the conversation.\n"
        "- Keep strings concise but meaningful (1–3 sentences for summary/support).\n"
    )

    instruction_text = (
        "I will now give you a JSON array called \"conversation_log\". Each element has:\n"
        "- \"role\": \"user\" or \"assistant\"\n"
        "- \"content\": the text\n\n"
        "Please:\n"
        "1. Read the entire conversation.\n"
        "2. Infer:\n"
        "   - What this conversation was mainly about (summary).\n"
        "   - The user’s dominant emotions.\n"
        "   - The main topics/themes.\n"
        "   - 1–3 short direct user quotes that capture the essence.\n"
        "   - A brief description of how Cara tried to support the user.\n"
        "   - 3–7 tags (short keywords) for retrieval.\n\n"
        "3. Output a single JSON object with EXACTLY the following keys:\n\n"
        "{\n"
        "  \"timestamp\": \"<leave this as an empty string \\\"\\\". The caller will fill it in.>\",\n"
        "  \"summary\": \"<1-3 sentences>\",\n"
        "  \"emotion\": \"<short description of main emotions, e.g. 'anxious, hurt'>\",\n"
        "  \"topics\": [\"<short topic 1>\", \"<short topic 2>\", \"...\"],\n"
        "  \"user_quotes\": [\"<short direct quote 1>\", \"<quote 2>\"],\n"
        "  \"cara_support\": \"<1-3 sentences describing how Cara responded / tried to help>\",\n"
        "  \"tags\": [\"<tag1>\", \"<tag2>\", \"<tag3>\"]\n"
        "}\n\n"
        "Important:\n"
        "- Your entire reply must be ONLY the JSON object, no backticks or extra text.\n"
        "- Keep topics and tags short, lowercase words or short phrases.\n"
        "- Prioritize emotional / personal content over technical small talk.\n\n"
        "Here is the conversation_log:\n"
        + json.dumps({"conversation_log": conversation_log}, ensure_ascii=False, indent=2)
    )

    raw_reply = call_llm_for_summary(system_prompt, instruction_text, conversation_log)

    # Try to parse JSON safely
    try:
        data = json.loads(raw_reply)
    except json.JSONDecodeError:
        # If the model returned extra text, try to salvage by finding the first '{' and last '}'
        start = raw_reply.find("{")
        end = raw_reply.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw_reply[start:end+1])
        else:
            raise

    # Now fill in the timestamp here
    ts = current_iso_timestamp()
    data["timestamp"] = ts

    # Map to the fields expected by save_session_entry
    summary = data.get("summary", "")
    emotion = data.get("emotion", "")
    topics = data.get("topics", [])
    user_quotes = data.get("user_quotes", [])
    cara_support = data.get("cara_support", "")
    tags = data.get("tags", [])

    # Save to a session file (this will also compute embeddings if you used that version)
    path = save_session_entry(
        summary=summary,
        emotion=emotion,
        topics=topics,
        user_quotes=user_quotes,
        cara_support=cara_support,
        tags=tags,
    )

    return path
