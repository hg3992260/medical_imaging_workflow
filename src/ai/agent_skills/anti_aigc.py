import logging
from collections import Counter

logger = logging.getLogger(__name__)

def _check_character_consistency(original: str, new: str, threshold: float = 0.1) -> bool:
    """
    Check if the new text has lost too many core characters (like 't', 'e', 'a').
    Returns False if the drop is greater than the threshold (e.g., > 10%).
    """
    if not original or not new:
        return True
        
    orig_counts = Counter(original.lower())
    new_counts = Counter(new.lower())
    
    core_chars = ['t', 'e', 'a', 'i', 'o', 'n']
    for char in core_chars:
        orig_c = orig_counts.get(char, 0)
        new_c = new_counts.get(char, 0)
        if orig_c > 20: # Only check if the character is somewhat frequent
            drop_rate = (orig_c - new_c) / float(orig_c)
            if drop_rate > threshold:
                logger.error(f"Consistency Check Failed: Character '{char}' dropped by {drop_rate:.1%} (from {orig_c} to {new_c})")
                return False
    return True

def humanize_skill(draft_text, llm_service, model):
    """
    Rewrite the input text to reduce its "AI-generated footprint" while preserving strict medical accuracy.
    """
    sys_prompt = """
    # Role
    You are a Senior Medical Writer who specializes in "Humanizing" technical drafts.

    # Task
    Rewrite the input text to reduce its "AI-generated footprint" while preserving strict medical accuracy.

    # Anti-AIGC Techniques to Apply:
    1. **Burstiness (Sentence Length Variation):** AI tends to write sentences of similar length. You MUST mix very short, punchy sentences with long, complex compound sentences.
    2. **Perplexity Injection (Vocabulary):** Replace common AI transition words (e.g., "Moreover", "Furthermore", "In conclusion") with more specific or abrupt academic transitions.
    3. **Structure Rotation:** Do not always start sentences with "The [Subject]...". Use introductory clauses, gerunds, or prepositional phrases.

    # Example Input
    "The study was conducted to evaluate the drug. The results showed a significant improvement. Therefore, we recommend this treatment." (Too Robotic)

    # Example Output
    "To evaluate the drug's efficacy, we conducted a rigorous study. Results were striking: a significant improvement was observed. Consequently, this treatment is highly recommended." (Higher Burstiness)
    
    CRITICAL: YOU MUST PRESERVE ALL ORIGINAL DATA AND TERMINOLOGY. DO NOT SHORTEN THE TEXT TOO MUCH. The output length should be roughly the same as the input length. Do not remove essential letters or words.
    
    RETURN ONLY THE REWRITTEN TEXT.
    """
    
    prompt = f"{sys_prompt}\n\n# Input Text:\n{draft_text}"
    
    res = llm_service.generate(model, prompt, timeout=600, options={"temperature": 0.7})
    if not res.get("success"):
        logger.warning(f"Humanize failed: {res.get('error')}")
        return draft_text
        
    rewritten_text = res.get("response", "").strip()
    
    # 字符丢失一致性检查 (Consistency Check)
    # 防止正则表达式误伤或者LLM幻觉导致的全局字符过滤
    if not _check_character_consistency(draft_text, rewritten_text, threshold=0.15):
        logger.error("Humanize aborted due to character consistency check failure. (Possible 't' dropping issue)")
        return draft_text
        
    return rewritten_text
