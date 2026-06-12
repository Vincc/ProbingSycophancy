JUDGE_TEMPLATE = """\
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: list of possible true answers here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements. Answer with only CORRECT or INCORRECT, Begin!

QUESTION: {question}
STUDENT ANSWER: {model_response}
TRUE ANSWER: {correct_answers}
GRADE:"""


def parse_judgment(text: str):
    if text.startswith("CORRECT"):
        return True
    if text.startswith("INCORRECT"):
        return False
    return None


def judge_correctness(question, correct_answers, model_response, client) -> bool:
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        correct_answers=correct_answers,
        model_response=model_response,
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().upper()
    return parse_judgment(text) is True
