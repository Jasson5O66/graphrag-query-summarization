import logging
from typing import List, Dict, Any
from .llm_handler import LLMHandler

# PROMPT for generating an answer from a single community summary
INTERMEDIATE_ANSWER_PROMPT = """
Based *only* on the following Community Summary, provide a concise answer to the User Query.
If the summary does not contain relevant information, respond with "This community does not provide relevant information."

User Query: {query}

Community Summary: {summary}

Answer:
"""

# PROMPT for synthesizing a final answer
FINAL_ANSWER_PROMPT = """
You are a research assistant. Synthesize a single, comprehensive, and well-structured final answer to the User's Global Query.
Use the provided list of "Intermediate Answers", each derived from a different text community.
Ignore any intermediate answers that state they are not relevant.
Combine the relevant information into one cohesive response.

User's Global Query: {query}

Intermediate Answers:
{intermediate_answers}

Final Comprehensive Answer:
"""

def generate_global_answer(
    query: str, community_summaries: List[Dict[str, Any]], llm: LLMHandler
) -> str:
    """
    Generates a global answer to a query by synthesizing answers from
    all community summaries.
    """
    logging.info(f"Generating global answer for query: '{query}'")
    
    intermediate_answers = []
    
    # 1. Get intermediate answers from each community
    logging.info(f"Generating intermediate answers for {len(community_summaries)} communities...")
    for i, community in enumerate(community_summaries):
        logging.info(f"Processing community {i+1}/{len(community_summaries)}...")
        prompt = INTERMEDIATE_ANSWER_PROMPT.format(
            query=query,
            summary=community['summary']
        )
        try:
            answer = llm.get_response(prompt, "")
            intermediate_answers.append(f"Community {i} Answer: {answer}")
            logging.debug(f"Community {i} intermediate answer: {answer}")
        except Exception as e:
            logging.error(f"Failed to get intermediate answer for community {i}: {e}")
            intermediate_answers.append(f"Community {i} Answer: Error processing.")
            
    logging.info("All intermediate answers generated.")
    
    # 2. Synthesize a final answer
    logging.info("Synthesizing final global answer...")
    intermediate_answers_text = "\n\n".join(intermediate_answers)
    
    final_prompt = FINAL_ANSWER_PROMPT.format(
        query=query,
        intermediate_answers=intermediate_answers_text
    )
    
    try:
        final_answer = llm.get_response(final_prompt, "")
        logging.info("Final answer synthesized.")
        return final_answer
    except Exception as e:
        logging.critical(f"Failed to synthesize final answer: {e}", exc_info=True)
        return "Error: Could not synthesize a final answer."
