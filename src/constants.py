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
    # Datasets that already contain instructions
    "ConvFinQA": None,
    "Headline": None,
    "FinNerCLS": None,
    "FinQA": None,
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
