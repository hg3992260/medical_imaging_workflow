import re
import dataclasses
from typing import List, Dict, Tuple

@dataclasses.dataclass
class FunnelIssue:
    layer: str        # Layer 1, 2, or 3
    severity: str     # BLOCK, BACKTRACK, DOWNGRADE, INFO
    message: str
    evidence: str = ""

@dataclasses.dataclass
class FunnelReport:
    original_text: str
    fixed_text: str
    issues: List[FunnelIssue]
    
    def has_blocks(self) -> bool:
        return any(i.severity == "BLOCK" for i in self.issues)
        
    def has_backtracks(self) -> bool:
        return any(i.severity == "BACKTRACK" for i in self.issues)
        
    def summary(self) -> str:
        if not self.issues:
            return "🎉 All Layers Passed! No issues detected."
        
        lines = []
        for issue in self.issues:
            lines.append(f"[{issue.layer}] {issue.severity}: {issue.message} {f'({issue.evidence})' if issue.evidence else ''}")
        return "\n".join(lines)


class FunnelLoopAnalyzer:
    """
    Three-Layer Funnel Loop for Academic Paper Quality Control
    """
    
    @classmethod
    def analyze_and_fix(cls, text: str) -> FunnelReport:
        issues = []
        
        # --- Pre-processing: Auto Fix Spelling ---
        fixed_text = text
            
        # --- Layer 1: Format & Completeness (BLOCK) ---
        l1_issues = cls._layer1_format_check(fixed_text)
        issues.extend(l1_issues)
        
        # --- Layer 2: Logic & Data Consistency (BACKTRACK) ---
        l2_issues = cls._layer2_logic_check(fixed_text)
        issues.extend(l2_issues)
        
        # --- Layer 3: Review Simulation & Downgrading (DOWNGRADE) ---
        l3_issues = cls._layer3_downgrade_check(fixed_text)
        issues.extend(l3_issues)
        
        return FunnelReport(original_text=text, fixed_text=fixed_text, issues=issues)

    @classmethod
    def _layer1_format_check(cls, text: str) -> List[FunnelIssue]:
        issues = []
        
        # Check 1: Long Placeholders or Comments (e.g., [TODO: ...], [Note to reviewer])
        long_placeholders = re.finditer(r"\[([^\]]{10,})\]", text)
        for m in long_placeholders:
            issues.append(FunnelIssue("Layer 1", "BLOCK", "检测到疑似长占位符或注释，需人工清理", m.group(0)))
            
        # Check 2: Self-correction indicators
        self_indicators = ["This revision", "Note to reviewer", "Self-correction", "Here is the rewritten text"]
        for ind in self_indicators:
            if ind.lower() in text.lower():
                issues.append(FunnelIssue("Layer 1", "BLOCK", "检测到 AI 自我指示语，请删除", ind))
                
        # Check 3: Empty bracket tags (e.g., [ ], [xxx] where xxx is not a digit)
        # Assuming citations are [1], [2,3]. If it's [Statistical Findings], it's a block.
        bad_brackets = re.finditer(r"\[([a-zA-Z\s_]+)\]", text)
        for m in bad_brackets:
            if len(m.group(1)) > 2:
                issues.append(FunnelIssue("Layer 1", "BLOCK", "检测到未替换的文本占位符", m.group(0)))
                
        return issues

    @classmethod
    def _layer2_logic_check(cls, text: str) -> List[FunnelIssue]:
        issues = []
        
        methods_match = re.search(r"Methods?(.*?)(?:Results?|Discussion|Conclusion)", text, re.IGNORECASE | re.DOTALL)
        results_match = re.search(r"Results?(.*?)(?:Discussion|Conclusion)", text, re.IGNORECASE | re.DOTALL)
        
        methods_text = methods_match.group(1) if methods_match else text
        results_text = results_match.group(1) if results_match else text

        # Check 1: Missing patient numbers
        if re.search(r"\bpatients?\b|\bparticipants?\b", methods_text, re.IGNORECASE):
            if not re.search(r"\b(?:n\s*=\s*\d+|\d+\s*patients?)\b", methods_text, re.IGNORECASE):
                issues.append(FunnelIssue("Layer 2", "BACKTRACK", "Methods 中提到患者，但未给出具体数量 (n=?)"))
                
        # Check 2: Missing p-value when claiming significance
        if re.search(r"\bsignificant(ly)?\b", results_text, re.IGNORECASE):
            if not re.search(r"p\s*[<>=]\s*0?\.\d+", results_text, re.IGNORECASE):
                issues.append(FunnelIssue("Layer 2", "BACKTRACK", "Results 中声明了'显著(significant)'，但未提供具体的 p 值"))
                
        # Check 3: Multiple dosage reduction percentages
        dose_reductions = re.findall(r"(\d{2,3})\s*%\s*(?:dose\s*reduction|reduction\s*in\s*dose)", text, re.IGNORECASE)
        if len(set(dose_reductions)) > 1:
            issues.append(FunnelIssue("Layer 2", "BACKTRACK", "检测到多个不同的剂量降低百分比，可能存在数值冲突", f"发现值: {', '.join(set(dose_reductions))}%"))

        return issues

    @classmethod
    def _layer3_downgrade_check(cls, text: str) -> List[FunnelIssue]:
        issues = []
        
        strong_claims = [
            (r"\bparadigm shift\b", "suggests a potential direction for future clinical implementation"),
            (r"\brevolutionary\b", "promising"),
            (r"\bclinically equivalent\b", "comparable under specific conditions"),
            (r"\bfirst[- ]ever\b", "novel"),
            (r"\bsignificantly superior\b", "demonstrated measurable improvements")
        ]
        
        for claim, downgrade_suggestion in strong_claims:
            for m in re.finditer(claim, text, re.IGNORECASE):
                issues.append(FunnelIssue("Layer 3", "DOWNGRADE", f"过度声称 (Over-claim): 建议降级替换为 '{downgrade_suggestion}'", m.group(0)))
                
        return issues
