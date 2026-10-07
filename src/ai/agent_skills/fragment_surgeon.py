import json
from src.services.ollama_local_service import OllamaLocalService
from src.utils.logger import get_logger

logger = get_logger(__name__)

def fragment_surgeon_skill(
    selected_text: str,
    pre_context: str,
    post_context: str,
    instruction: str,
    data_context: str,
    llm_service: OllamaLocalService,
    model: str = "llama3"
) -> str:
    """
    Refines a text fragment based on user instruction and specific data context.
    
    Args:
        selected_text: The text to refine.
        pre_context: Text preceding the selection.
        post_context: Text following the selection.
        instruction: User's refinement instruction.
        data_context: Stringified data (e.g., JSON or text) to use as ground truth.
        llm_service: Service to call LLM.
        model: Model to use.
        
    Returns:
        Refined text fragment.
    """
    
    system_prompt = (
        "You are a precise scientific editor. Your task is to rewrite the `Selected Text` "
        "based on the `User Instruction` and the provided `Strict Data Context`.\n"
        "Rules:\n"
        "1. You must verify any numbers or claims against the `Strict Data Context`.\n"
        "2. You must ensure the rewritten text flows grammatically with `Pre Context` and `Post Context`.\n"
        "3. Do NOT invent information not present in the data.\n"
        "4. Return ONLY the rewritten text for the `Selected Text` part. Do not include pre/post context in the output unless necessary for transition.\n"
        "5. Do not include markdown or explanations."
    )
    
    user_prompt = (
        f"Pre Context: ...{pre_context[-500:]}\n\n"
        f"Selected Text: {selected_text}\n\n"
        f"Post Context: {post_context[:500]}...\n\n"
        f"User Instruction: {instruction}\n\n"
        f"Strict Data Context:\n{data_context}\n\n"
        "Rewritten Text:"
    )
    
    try:
        response = llm_service.generate(model, system_prompt + "\n\n" + user_prompt)
        if response['success']:
            return response['response'].strip()
        else:
            logger.error(f"Fragment Surgeon failed: {response.get('error')}")
            return selected_text # Fallback
    except Exception as e:
        logger.error(f"Fragment Surgeon exception: {e}")
        return selected_text
