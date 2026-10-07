import difflib
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def _check_character_consistency(original: str, new: str, threshold: float = 0.15) -> bool:
    if not original or not new:
        return True
    orig_counts = Counter(original.lower())
    new_counts = Counter(new.lower())
    core_chars = ['t', 'e', 'a', 'i', 'o', 'n']
    for char in core_chars:
        orig_c = orig_counts.get(char, 0)
        new_c = new_counts.get(char, 0)
        if orig_c > 20:
            drop_rate = (orig_c - new_c) / float(orig_c)
            if drop_rate > threshold:
                logger.error(f"Grammar Consistency Check Failed: Character '{char}' dropped by {drop_rate:.1%}")
                return False
    return True

def grammar_check_skill(draft_text, llm_service, model):
    """
    使用 Ollama 进行仅限于语法和拼写的修正，严禁改写意思。
    """
    sys_prompt = """
    You are a professional Academic Copy Editor. 
    Your ONLY task is to fix grammatical errors, typos, and punctuation mistakes in the provided text.
    1. DO NOT change the meaning.
    2. DO NOT change medical terminology (ICD/SNOMED codes).
    3. DO NOT shorten or summarize. 
    4. CRITICAL: PRESERVE ALL CHARACTERS AND WORDS that are correct. Do not drop letters.
    5. Output ONLY the corrected text.
    """
    
    prompt = f"System: {sys_prompt}\n\nUser: {draft_text}"
    
    # Use existing llm_service.generate with infinite timeout
    res = llm_service.generate(model, prompt, timeout=600)
    if not res.get("success"):
        logger.warning(f"Grammar check failed: {res.get('error')}")
        return draft_text
        
    corrected_text = res.get("response", "").strip()
    if not corrected_text:
        logger.warning("Grammar checker returned empty content. Keeping current draft.")
        return draft_text
        
    if not _check_character_consistency(draft_text, corrected_text):
        logger.error("Grammar check aborted due to character consistency check failure. (Possible 't' dropping issue)")
        return draft_text
    
    # 计算差异率，如果改动过大（比如超过 10%），说明模型可能在篡改内容，需要报警
    diff_ratio = difflib.SequenceMatcher(None, draft_text, corrected_text).ratio()
    if diff_ratio < 0.97:
        logger.warning(f"⚠️ Warning: Grammar checker modified heavily ({1-diff_ratio:.2%}). Reverting to original text.")
        return draft_text
        
    return corrected_text
