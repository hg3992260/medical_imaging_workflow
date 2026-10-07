import logging
import json
from src.utils.local_template_manager import LocalTemplateManager

logger = logging.getLogger(__name__)

class TemplateFitter:
    """
    Agent Skill: Template Fitting (TemplateFit Wizard)
    Adapts the draft content to specific journal requirements (Word count, Structure, Citation style).
    """

    def __init__(self, llm_service, model="llama3"):
        self.llm_service = llm_service
        self.model = model
        self.local_manager = LocalTemplateManager()

    def search_online_templates(self, query: str):
        """
        Mock online search for journal templates.
        In a real scenario, this would call a web search API (e.g., SerpApi, Bing) 
        and then use LLM to parse the 'Instructions for Authors' page.
        """
        logger.info(f"Searching online for templates matching: {query}")
        
        # Check local first (fuzzy search)
        local_result = self.local_manager.search_local(query)
        if local_result:
            return {query: local_result} # Wrap in dict to match expected return format

        # Mock Search Results (Simulating what an LLM would extract from web pages)
        # This allows us to "simulate" finding new templates dynamically.
        mock_results = {}
        
        if "radiology" in query.lower():
            mock_results["Radiology"] = {
                "journal_name": "Radiology (RSNA)",
                "max_word_count": {"abstract": 300, "main_text": 3000},
                "structure": ["Summary", "Introduction", "Materials and Methods", "Results", "Discussion"],
                "citation_style": "AMA"
            }
        elif "lancet" in query.lower():
            mock_results["The Lancet"] = {
                "journal_name": "The Lancet",
                "max_word_count": {"abstract": 250, "main_text": 3000},
                "structure": ["Introduction", "Methods", "Findings", "Interpretation"],
                "citation_style": "Vancouver"
            }
        elif "ieee" in query.lower():
             mock_results["IEEE TMI"] = {
                "journal_name": "IEEE Trans. Med. Imaging",
                "max_word_count": {"abstract": 250, "main_text": 5000},
                "structure": ["Introduction", "Methods", "Experiments", "Conclusion"],
                "citation_style": "IEEE"
            }
            
        return mock_results

    def get_template_constraints(self, template_name: str):
        """
        Retrieves constraints. Checks local cache first, then 'online' mock.
        """
        # 1. Local Cache (Local Templates)
        local_config = self.local_manager.search_local(template_name)
        if local_config:
            # If it's a local template, we might need to load constraints from a config inside the template folder
            # or just use what's in registry.json.
            # Assuming registry.json has the constraints or minimal info.
            # For now, let's assume registry has enough info or we mock it if missing.
            if "journal_name" not in local_config:
                local_config["journal_name"] = template_name
            return local_config

        # 2. Hardcoded Common Templates
        templates = {
            "NEJM": {
                "journal_name": "New England Journal of Medicine",
                "max_word_count": {"abstract": 250, "main_text": 2700},
                "structure": ["Introduction", "Methods", "Results", "Discussion"],
                "citation_style": "Vancouver (Numeric)"
            },
            "JAMA": {
                "journal_name": "Journal of the American Medical Association",
                "max_word_count": {"abstract": 300, "main_text": 3000},
                "structure": ["Importance", "Objective", "Design", "Setting", "Participants", "Main Outcomes", "Results", "Conclusions"],
                "citation_style": "AMA"
            },
            "Nature": {
                "journal_name": "Nature",
                "max_word_count": {"abstract": 150, "main_text": 2000}, # Letter format
                "structure": ["Abstract (Unstructured)", "Main Text", "Methods"],
                "citation_style": "Nature (Superscript)"
            }
        }
        
        # 3. Check hardcoded
        if template_name in templates:
            return templates[template_name]
            
        # 4. If not, try to "search" (Mock fallback for demo)
        # In a real app, we would have stored the result of search_online_templates into a persistent cache.
        # Here we just re-run the mock logic to return data if it matches our mock keywords.
        online = self.search_online_templates(template_name)
        if online:
            # Return the first match's values
            return list(online.values())[0]
            
        return templates["NEJM"] # Default fallback

    def fit_draft(self, draft_text: str, template_name: str):
        """
        Main entry point for template fitting loop.
        """
        constraints = self.get_template_constraints(template_name)
        logger.info(f"Fitting draft to {constraints['journal_name']} constraints...")
        
        # 1. Plan Phase
        plan = self._plan_modifications(draft_text, constraints)
        
        # 2. Execution Phase (Refiner)
        fitted_text = self._execute_fitting(draft_text, plan, constraints)
        
        # 3. Validation Phase (Simple Check)
        validation = self._validate_fitting(fitted_text, constraints)
        
        return {
            "fitted_text": fitted_text,
            "constraints": constraints,
            "plan": plan,
            "validation": validation
        }

    def _plan_modifications(self, text, constraints):
        """
        Use LLM to plan how to adapt the text.
        """
        prompt = (
            f"You are a Medical Editor. Plan how to adapt the following draft to the target journal requirements.\n"
            f"Target Journal: {constraints['journal_name']}\n"
            f"Constraints: {json.dumps(constraints, indent=2)}\n\n"
            f"Draft Text Summary (First 500 chars): {text[:500]}...\n\n"
            f"Task: List 3-5 specific structural or stylistic changes needed. Return ONLY the list."
        )
        res = self.llm_service.generate(self.model, prompt)
        return res.get("response", "No plan generated.")

    def _execute_fitting(self, text, plan, constraints):
        """
        Use LLM to rewrite the text according to the plan.
        """
        prompt = (
            f"You are a Medical Editor. Rewrite the provided draft to strictly follow the Target Journal constraints.\n"
            f"Target Journal: {constraints['journal_name']}\n"
            f"Constraints: {json.dumps(constraints, indent=2)}\n"
            f"Modification Plan: {plan}\n\n"
            f"Draft Text:\n{text}\n\n"
            f"Instructions:\n"
            f"1. Adjust structure/headings to match constraints.\n"
            f"2. Ensure word count is close to limits (summarize if needed).\n"
            f"3. Return the FULL rewritten text."
        )
        # Higher timeout for full rewrite (Infinite)
        res = self.llm_service.generate(self.model, prompt, timeout=None)
        return res.get("response", text) # Fallback to original

    def _validate_fitting(self, text, constraints):
        """
        Simple validation logic.
        """
        issues = []
        # Check word count (Approximate)
        words = len(text.split())
        limit = constraints.get("max_word_count", {}).get("main_text", 3000)
        if words > limit * 1.1: # 10% tolerance
            issues.append(f"Word count ({words}) exceeds limit ({limit})")
            
        return {"status": "PASS" if not issues else "WARN", "issues": issues}
