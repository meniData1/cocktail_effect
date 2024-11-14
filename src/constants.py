META_PROMPTS_DICT = {
    "FPB": (
        "### Instruction:\n"
        "You are given a financial document. Your task is to infer its sentiment.\n"
        "Answer using one of the following labels: ['Negative', 'Neutral', 'Positive'], and include nothing else.\n"
        "You must answer with a single word, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
        "### Response:\n"
    ),
    "Financebench": (
        "### Instruction:\n"
        "You are given a financial question and a a financial document. Your task is to answer the question based on the document.\n"
        "### Document:\n"
        "\n{document}\n"
        "### Question:\n"
        "\n{question}\n"
        "### Response:\n"
    ),  
    "FiQA_SA": (
        "### Instruction:\n"
        "You are given a financial sentence. Your task is to infer its sentiment.\n"
        "Answer using one of the following labels: ['Negative', 'Neutral', 'Positive'], and include nothing else.\n"
        "You must answer with a single word, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
        "### Response:\n"
    ),
    "Twitter_SA": (
        "### Instruction:\n"
        "You are given a financial sentence taken from Twitter. Your task is to infer its sentiment.\n"
        "Answer using one of the following labels: ['Bearish', 'Bullish', 'Neutral'], and include nothing else.\n"
        "You must answer with a single word, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
        "### Response:\n"
    ),
    "Twitter_Topics": (
        "### Instruction:\n"
        "You are given a financial sentence taken from Twitter. Your task is to infer its topic.\n"
        "Answer using one of the following labels: {labels}"
        ", and include nothing else.\n"
        "You must answer with a single topic, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
        "### Response:\n"
    ),
    "CFA": (
        "### Instruction:\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the three options A, B and C."
        "You must answer with a single letter corresponding to the correct answer, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "CFA_COT": (
        "### Instruction:\n"
        "You are a financial expert, tasked with answering multiple-choice questions taken from the CFA exam.\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the three options A, B and C."
        "Think step-by-step, and describe your reasoning process clearly before providing the final answer."
        "You must provide the correct answer in a clear manner.\n"
        "Begin by describing your detailed reasoning process in a step-by-step manner, and then provide the final answer.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "CPA": (
        "### Instruction:\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the four options A, B, C, and D."
        "You must answer with a single letter corresponding to the correct answer, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "CPA_COT": (
        "### Instruction:\n"
        "You are a financial expert, tasked with answering multiple-choice questions taken from the CPA exam.\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the four options A, B, C, and D."
        "Think step-by-step, and describe your reasoning process clearly before providing the final answer."
        "You must provide the correct answer in a clear manner.\n"
        "Begin by describing your detailed reasoning process in a step-by-step manner, and then provide the final answer.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "IMC": (
        "### Instruction:\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the four options a, b, c and d."
        "You must answer with a single letter corresponding to the correct answer, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "IMC_COT": (
        "### Instruction:\n"
        "You are a financial expert, tasked with answering multiple-choice questions taken from the IMC exam.\n"
        "Read the questions and answers carefully, and choose the one you think is appropriate among the four options a, b, c and d."
        "Think step-by-step, and describe your reasoning process clearly before providing the final answer."
        "You must provide the correct answer in a clear manner.\n"
        "Begin by describing your detailed reasoning process in a step-by-step manner, and then provide the final answer.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "IMC_Numerical": (
        "### Instruction:\n"
        "You are a financial expert, tasked with answering numerical questions taken from the IMC exam.\n"
        "Read the question carefully, and provide the correct numerical answer.\n"
        "You must provide the correct numerical answer, and no additional context.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "IMC_Numerical_COT": (
        "### Instruction:\n"
        "You are a financial expert, tasked with answering numerical questions taken from the IMC exam.\n"
        "Read the question carefully, and provide the correct numerical answer.\n"
        "Think step-by-step, and describe your reasoning process clearly before providing the final answer."
        "You must provide the correct answer in a clear manner.\n"
        "Begin by describing your detailed reasoning process in a step-by-step manner, and then provide the final answer.\n"
        "### Input:\n"
        "\n{document}\n"
    ),
    "TatQA": (
        "### Instruction:\n"
        "You are given a table, the text that was provided with it, and a question. Your task is to answer the question.\n"
        "A question may have multiple answers, and you must provide all of them.\n"
        "You must provide only the requested information.\n"
        "Make sure to return the answers in the correct order.\n"
        "You must return a list containing the relevant answers and nothing else. The format should be:\n"
        "['answer1', 'answer2', ...]\n"
        "You must answer with a single list of answers in that format, and nothing else.\n"
        "### Input:\n"
        "\n{document}\n"
        "### Response:\n"
    ),
    # Datasets that already contain instructions
    "ConvFinQA": None,
    "Headline": None,
    "FinNerCLS": None,
    "FinQA": None,
    "TACT": None,
    "TactText": None,
}

NUM_TO_LABEL_DICT = {
    "Twitter_SA": {0: "Bearish", 1: "Bullish", 2: "Neutral"},
    "Twitter_Topics": {
        0: "Analyst Update",
        1: "Fed | Central Banks",
        2: "Company | Product News",
        3: "Treasuries | Corporate Debt",
        4: "Dividend",
        5: "Earnings",
        6: "Energy | Oil",
        7: "Financials",
        8: "Currencies",
        9: "General News | Opinion",
        10: "Gold | Metals | Materials",
        11: "IPO",
        12: "Legal | Regulation",
        13: "M&A | Investments",
        14: "Macro",
        15: "Markets",
        16: "Politics",
        17: "Personnel Change",
        18: "Stock Commentary",
        19: "Stock Movement",
    },
}
