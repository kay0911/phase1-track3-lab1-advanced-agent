from __future__ import annotations
import os
import re
import json
import time
import urllib.request
import urllib.error
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

# Load environment variables
load_dotenv()

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# Metrics tracker for real LLM mode
METRICS = {
    "tokens": 0,
    "latency_ms": 0
}

def reset_metrics():
    METRICS["tokens"] = 0
    METRICS["latency_ms"] = 0

def should_run_real() -> bool:
    return os.getenv("MOCK_MODE", "true").lower() == "false"
API_KEYS = None
LAST_CALL_TIMES = {}
CURRENT_KEY_INDEX = 0

def get_api_keys() -> list[str]:
    global API_KEYS, LAST_CALL_TIMES
    if API_KEYS is None:
        keys_env = os.getenv("GEMINI_API_KEYS")
        if keys_env:
            API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
        else:
            single_key = os.getenv("GEMINI_API_KEY")
            if single_key:
                API_KEYS = [single_key]
            else:
                API_KEYS = []
        LAST_CALL_TIMES = {key: 0.0 for key in API_KEYS}
    return API_KEYS

def call_gemini_api(system_prompt: str, user_prompt: str, response_json: bool = False) -> str:
    global CURRENT_KEY_INDEX
    keys = get_api_keys()
    if not keys:
        raise ValueError("GEMINI_API_KEYS/GEMINI_API_KEY not found in environment variables. Make sure MOCK_MODE is true if you want to run without Gemini API.")
        
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    contents = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_prompt
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        }
    }
    
    if response_json:
        contents["generationConfig"] = {
            "responseMimeType": "application/json"
        }
        
    data = json.dumps(contents).encode("utf-8")
    
    # Retry loop (up to 3 attempts)
    for attempt in range(3):
        # Pick the key and rotate
        key = keys[CURRENT_KEY_INDEX]
        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(keys)
        
        # Rate limiting: ensure at least 4.2 seconds between API calls FOR THIS KEY
        now = time.time()
        elapsed = now - LAST_CALL_TIMES.get(key, 0.0)
        if elapsed < 4.2:
            time.sleep(4.2 - elapsed)
            
        LAST_CALL_TIMES[key] = time.time()
        start_time = time.time()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as response:
                res_body = response.read().decode("utf-8")
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                res_json = json.loads(res_body)
                text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                
                # Update METRICS
                usage = res_json.get("usageMetadata", {})
                tokens = usage.get("totalTokens", 0)
                if tokens == 0:
                    prompt_tokens = usage.get("promptTokenCount", 0)
                    candidates_tokens = usage.get("candidatesTokenCount", 0)
                    tokens = prompt_tokens + candidates_tokens
                
                METRICS["tokens"] += tokens
                METRICS["latency_ms"] += latency_ms
                
                return text
        except urllib.error.HTTPError as e:
            try:
                err_msg = e.read().decode("utf-8")
            except Exception:
                err_msg = "Unknown error details"
            print(f"Gemini API HTTP Error (Attempt {attempt+1}/3 with key index {CURRENT_KEY_INDEX}): {e.code} - {err_msg}")
            if e.code == 429:
                time.sleep(5)
                continue
            if attempt == 2:
                raise e
        except Exception as e:
            print(f"Network error (Attempt {attempt+1}/3): {str(e)}")
            if attempt == 2:
                raise e
            time.sleep(2)
            
    raise RuntimeError("Failed to call Gemini API after 3 attempts.")

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if not should_run_real():
        if example.qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[example.qid]
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[example.qid]
        return example.gold_answer
        
    # Real LLM mode
    context_str = "\n\n".join(f"Title: {c.title}\nText: {c.text}" for c in example.context)
    reflections_str = "\n".join(f"Attempt {i+1}: {ref}" for i, ref in enumerate(reflection_memory)) if reflection_memory else "No previous attempts."
    
    user_prompt = f"Ngữ cảnh (Context):\n{context_str}\n\nNhật ký phản chiếu (Reflection Memory):\n{reflections_str}\n\nCâu hỏi: {example.question}"
    
    response_text = call_gemini_api(ACTOR_SYSTEM, user_prompt, response_json=False)
    
    # Extract answer using regex [ANSWER]...[/ANSWER]
    match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback to look for "Final Answer:"
    match_fa = re.search(r"final answer:\s*(.*)", response_text, re.IGNORECASE)
    if match_fa:
        return match_fa.group(1).strip()
        
    return response_text.strip()

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if not should_run_real():
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])
        
    # Real LLM mode
    user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"
    
    response_text = call_gemini_api(EVALUATOR_SYSTEM, user_prompt, response_json=True)
    
    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        data = json.loads(clean_text)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", "No reason provided."),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", [])
        )
    except Exception as e:
        print(f"Error parsing evaluator JSON: {e}. Raw response: {response_text}")
        score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
        reason = "Matches gold answer (fallback)" if score == 1 else "Does not match gold answer (fallback)"
        return JudgeResult(score=score, reason=reason)

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if not should_run_real():
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
        
    # Real LLM mode
    user_prompt = f"Question: {example.question}\nWrong Answer: {judge.spurious_claims or 'N/A'}\nEvaluation Reason: {judge.reason}"
    
    response_text = call_gemini_api(REFLECTOR_SYSTEM, user_prompt, response_json=True)
    
    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        data = json.loads(clean_text)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson=data.get("lesson", "Verify findings carefully."),
            next_strategy=data.get("next_strategy", "Do multi-hop reasoning step-by-step.")
        )
    except Exception as e:
        print(f"Error parsing reflector JSON: {e}. Raw response: {response_text}")
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Failed to parse reflection JSON.",
            next_strategy="Carefully check context paragraphs for the missing entities."
        )
