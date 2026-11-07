# How large language models work

A large language model (LLM) is a neural network trained to predict the next token in a sequence. A token is a piece of text, often a word or part of a word. Models read and write tokens, so prices, speed and limits are all measured in tokens.

Most modern LLMs use the transformer architecture, introduced in the 2017 paper "Attention Is All You Need". Its key idea is self-attention: every token can look at every other token in the input and weigh how relevant each one is. This lets the model connect ideas that are far apart in a long document.

The context window is the maximum number of tokens a model can consider at once, including the prompt, any documents and its own reply. Anything outside the context window is invisible to the model, which is why retrieval and memory systems exist.

Temperature controls randomness when the model picks the next token. A temperature near zero makes answers more predictable, which suits agents, extraction and evaluations. Higher temperatures give more varied, creative text.

LLMs can produce fluent text that is wrong, known as hallucination. Grounding the model in retrieved sources, asking it to cite them and checking answers with evaluations all reduce this risk.
