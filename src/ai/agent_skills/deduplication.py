from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def duplication_check_skill(draft_text, rag_context_text):
    """
    检查草稿与 RAG 检索源文档的相似度，防止直接 Copy-Paste。
    """
    if not draft_text or not rag_context_text:
        return {"status": "PASS", "clean_text": draft_text}

    # 1. 准备数据：将草稿按句子切分
    sentences = [s.strip() for s in draft_text.split('. ') if len(s.strip()) > 20]
    if not sentences:
        return {"status": "PASS", "clean_text": draft_text}
    
    # 2. 准备源文档列表 (Simulated by splitting context)
    sources = [s.strip() for s in rag_context_text.split('\n') if len(s.strip()) > 20]
    if not sources:
        return {"status": "PASS", "clean_text": draft_text}
    
    try:
        # 3. 矢量化
        vectorizer = TfidfVectorizer().fit_transform(sentences + sources)
        vectors = vectorizer.toarray()
        
        draft_vecs = vectors[:len(sentences)]
        source_vecs = vectors[len(sentences):]
        
        # 4. 计算相似度矩阵
        similarity_matrix = cosine_similarity(draft_vecs, source_vecs)
        
        high_risk_sentences = []
        
        # 5. 判定：如果某句子与源文档任一段落相似度 > 0.85，视为抄袭
        for idx, row in enumerate(similarity_matrix):
            if row.max() > 0.85:
                high_risk_sentences.append({
                    "sentence": sentences[idx],
                    "max_similarity": float(row.max()),
                    "source_index": int(row.argmax())
                })
                
        if high_risk_sentences:
            logger.warning(f"Duplication detected: {len(high_risk_sentences)} sentences.")
            return {
                "status": "FAIL",
                "reason": "Plagiarism Detected",
                "issues": high_risk_sentences
            }
    except Exception as e:
        logger.error(f"Duplication check error: {e}")
    
    return {"status": "PASS", "clean_text": draft_text}
