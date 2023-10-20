---
layout: post
title: LLMs (Text Generative AI) Evaluations and Metrics
listing: Machine Learning Blogs
---


Language Models, often referred to as Large Language Models (LLMs), have revolutionized the way computers understand and generate human language. They're like super-smart AI writers, capable of crafting coherent and context-aware text. In this blog, we're diving into the world of LLMs, text generation models (Next token prediction) and its evaluation techiniques.

![]({{ site.baseurl }}/images/llm_metrics.png "LLM perplexity calculation")

Why is this important?

Well, understanding how to evaluate and fine-tune these models for custom tasks is like having a magic wand for creating human-like text tailored to your needs. We'll explore the metrics and techniques needed to master for harnessing the full potential of LLMs in our own projects. Evaluating and measuring the performance of Language Model text generation is crucial for assessing their quality and effectiveness.


Here's the table of contents:

1. TOC
{:toc}


# Introduction to LLMs Tasks
LLMs find their way into various real-world applications. Take ChatGPT as an example – it's like a friendly AI companion, making chatbots smarter and more engaging and solving problems across domains. Then there's Copilot, which helps programmers write code faster. These LLMs are versatile problem-solvers that improve communication, creativity, and efficiency in a wide range of tasks.

It's truly fascinating to learn about how these trained models are compared automatically. Evaluating LLMs like ChatGPT and Copilot is a complex challenge. They need to be assessed for fluency, relevance, correctness, and context, but they also must understand nuance, tone, and cultural sensitivities, which can be quite tricky. Metrics like BLEU and ROUGE provide some insight, but capturing the richness of human language remains elusive.

#### Text Generation Models Use Cases and Domains:
   - Text summarization
   - Machine translation (One language to other language)
   - Conversational agents (Chatbots)
   - Content generation
   - Creative writing
   - Code generation
   - Audio transcription
   - Parapharsing (Same meaning with different words)
   - Question Answering
   - Image captioning

The metrics for LLMs depend greatly on the specific task at hand, as different applications have unique evaluation needs. For ChatGPT, important metrics include fluency, coherence, and appropriateness. These models are often measured using standard language evaluation metrics like BLEU and perplexity, but task-specific criteria are essential for more nuanced understanding and improvement. For example Copilot is trained to generate code and BLEU has problems capturing semantic features specific to code so they used metric called pass@k. [paper link](https://arxiv.org/pdf/2107.03374.pdf)


# Automatic vs Human evaluation

In the world of Language Models, the evaluation process often hinges on two key approaches: automatic and human evaluation. Automatic evaluation involves using metrics like BLEU and ROUGE to quantitatively assess generated text, which is efficient but can miss nuanced language aspects. On the other hand, human evaluation relies on human judges to provide qualitative feedback, capturing more subtle aspects like coherence and appropriateness. Striking the right balance between these two approaches is vital for creating more robust and contextually aware LLMs. For ChatGPT, pretrained models is used to finetune on reward system to optimize it for human style dialogs this technique is called RLHF (Reinforcement Learning from Human Feedback).

# BLEU (Bilingual Evaluation Understudy)

BLEU is a metric for automatically evaluating machine generated text. It quantifies how well the generated text aligns with a reference or human-generated text.

It calculates a score based on the precision of n-grams (sequences of n words) in the generated text compared to the reference. The precision measures how many of the n-grams in the generated text overlap with those in the reference text. BLEU considers different n-gram lengths, typically from unigrams (single words) to four-grams, to capture both lexical and structural aspects.

Mathematically:

$$
\mathrm{BLEU}=\underbrace{\min \left(1, \exp \left(1-\frac{\text { reference-length }}{\text { output-length }}\right)\right)}_{\text {brevity penalty }} \underbrace{\left(\prod_{i=1}^4 \text { precision }_i\right)^{1 / 4}}_{\text {n-gram overlap }}
$$

$$
\text { precision }_i=\frac{\sum_{\mathrm{snt} \in \text { Cand-Corpus }} \sum_{i \in \mathrm{snt}} \min \left(m_{\text {cand }}^i, m_{\text {ref }}^i\right)}{w_t^i=\sum_{\text {snt }^{\prime} \in \text { Cand-Corpus }} \sum_{i^{\prime} \in \mathrm{snt}^{\prime}} m_{\text {cand }}^{i^{\prime}}}
$$

where, 

$$ m_{\text {cand }}^i $$: is the count of i-gram in candidate matching the reference translation

$$ m_{\text {ref }}^i $$: is the count of i-gram in the reference translation

$$ w_t^i $$: is the total number of i-grams in candidate translation

The brevity penalty penalizes generated translations that are too short compared to the closest reference length with an exponential decay. The brevity penalty compensates for the fact that the BLEU score has no recall term. BLEU score is in the range of [0, 1] where score > 0.6 means Better than human quality and < 0.1 means Useless.

```python
from nltk.translate.bleu_score import corpus_bleu
references = [[
    ['this', 'is', 'a', 'test'],
    ['this', 'is' 'test']
]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print(score)
# => 1
```

While BLEU is widely used for automatic evaluation, it has limitations. It may not account for fluency, coherence, or paraphrasing, and it's sensitive to minor variations in word order. 

# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE calculates several precision-based scores by examining various units of text, such as unigrams (single words) and n-grams (n-word sequences). It also considers the recall, which evaluates how well the generated text captures information from the reference text. Similar to BLEU it measures the overlap between the generated text and reference or human-produced text. In rouge we calculate both precision and recall and do F1 score unlike BLEU only prefers percision.

There are 3 types:
- n-rouge (1-rouge[1-gram] and 2-rouge[2-gram]),
- l-rouge: langest common subsequence
- s-rouge (skip n-gram).

```python
import evaluate
rouge = evaluate.load('rouge')
predictions = ["Transformers Transformers are fast plus efficient", 
               "Good Morning", "I am waiting for new Transformers"]
references = [
              ["HuggingFace Transformers are fast efficient plus awesome", 
               "Transformers are awesome because they are fast to execute"], 
              ["Good Morning Transformers", "Morning Transformers"], 
              ["People are eagerly waiting for new Transformer models", 
               "People are very excited about new Transformers"]

]
results = rouge.compute(predictions=predictions, references=references)
print(results)

# => {'rouge1': 0.6659340659340659, 'rouge2': 0.45454545454545453, 
# 'rougeL': 0.6146520146520146, 'rougeLsum': 0.6146520146520146}
```

ROUGE is particularly useful for evaluating text summarization, as it can assess the effectiveness of summarization systems in retaining important content from the source text. It is valuable for content-based evaluation but, like other automatic metrics, may not fully capture aspects of fluency, coherence, and overall human understanding, which often necessitates the inclusion of human evaluation for a more comprehensive assessment of text generation quality.

# Perplexity

Perplexity is a metric commonly used to evaluate the quality and fluency of language models. It quantifies how well a language model predicts a given sentence. It dosn't need reference text. The lower the perplexity score, the better the model is at predicting the text it encounters.

Mathematically, 
Likelihood of sequence is product of all token probabilities

$$ P(X)=\prod_{i=0}^t p\left(x_i \mid x_{<i}\right) $$

from Likelihood we can calculate Cross Entroy  $$ CE(X) $$ which is used in many tasks as a loss function.

$$ CE(X)=-\frac{1}{t} \log P(X) $$

by exponantiating the cross entropy we get perplexity

$$
\begin{aligned}
\operatorname{PPL}(X) & =e^{C E(X)} \\
& =e^{-\frac{1}{t} \sum_{i=0}^t \log p\left(x_i \mid x_{<i}\right)}
\end{aligned}
$$

It it means perplexity is closely related to the loss. Loss is a weak proxy of ability to generate quality text and the same is true for perplexity.

Lower perplexity indicates a better-fitting model, as it means the model's predictions align well with the actual text. A good language model should have a low perplexity score, reflecting its ability to accurately predict words in a given context.

# Using LLMs for Evaluation

Using a Large Language Model (LLM) to evaluate another LLM is a powerful method to assess and improve text generation quality. By employing a pre-trained LLM, you can create a benchmark for generated content. You then measure the similarity, coherence, and relevance of the generated text to this benchmark. The LLM can help detect issues like lack of context awareness, coherence, or overuse of certain phrases. 

Fine-tuning the model based on these evaluations allows for a significant enhancement in text quality, helping LLMs to better understand context and generate more relevant and human-like content across various applications, from chatbots to content generation. Below is the example code which leverage this method in question answering task using langchain.

```python
from langchain.evaluation.qa import QAEvalChain

predictions = [{
    'query': 'Is sea water salted?' # user query
    'answer': 'Yes' # reference by human
    'result': 'Sea water has high concentraction of salt in it' # model generated text
}]

llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)

print(graded_outputs[i]['text'])
=> CORRECT
```

# Pass@k Metric

Pass@k metrics is used in evaluating code generation models like Codex(backbone of Copilot). Match-based metrics are unable to account for the large and complex space of programs functionally. As a consequence, pseudocode-to-code translation uses functional correctness. where a sample is considered correct if it passes a set of unit tests.

Using the pass@k metric, where k code samples are generated per problem, a problem is considered solved if any sample passes the unit tests, and the total fraction of problems solved is reported.

However, computing pass@k in this way can have high variance. Instead, to evaluate pass@k, we generate n ≥ k samples per task count the number of correct samples c ≤ n which pass unit tests, and calculate the unbiased estimator

$$
\text { pass@ } k:=\underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{\left(\begin{array}{c}
n-c \\
k
\end{array}\right)}{\left(\begin{array}{l}
n \\
k
\end{array}\right)}\right]
$$

# METEOR (Metric for Evaluation of Translation with Explicit Ordering)

METEOR is a metric used to evaluate the quality of machine translation systems. The METEOR score is calculated using a formula that considers precision, recall, stemming, synonymy, and word order. Here's the mathematical formula for METEOR:

$$ \text{METEOR} = \frac{10PR}{(9P + R)} - β * (1 - p) $$

where,

$$ p=0.5\left(\frac{c}{u_m}\right)^3 $$

In this formula:
- c number of chunks in the candidate (take alignment with the fewest no of chunks).
- $$u_m$$ unigram in the candidate.
- p chunk penalty.
- P represents precision.
- R represents recall.
- β is a parameter used to adjust the penalty for incorrect word order usually 0.5.
- The Penalty term accounts for word stems, synonyms, and word order differences.

METEOR not only measures the overlap between generated and reference text but also considers synonyms and word order. Recall & Precision can be calculated using various techniques like semantic similarity(embedding based) or synonyms with stemmer. This makes it a more robust metric for assessing the fluency, adequacy, and grammatical correctness of machine-generated translations, offering a richer evaluation of translation quality.

# LSA (Latent Semantic Analysis)

LSA uncovers the hidden, or latent, relationships between words in a large corpus of text. It achieves this by representing words and documents in a high-dimensional space and reducing the dimensions (technique similar to PCA) to capture the underlying semantic structures. This process helps discover word associations and semantic similarities, making it valuable for tasks like document clustering, information retrieval, and topic modeling.

LSA can be used to evaluate LLMs by comparing the semantic similarity between the model-generated text and reference text. By applying LSA, one can assess how well LLMs capture the underlying semantics of the text, providing insights into the quality of generated content in terms of coherence and relevance to the reference, thereby aiding in evaluation.

# WER (Word Error Rate)

Word Error Rate (WER) is a metric used to evaluate the accuracy of automatic speech recognition (ASR) systems or transcription services in converting spoken language into written text. WER calculates the rate of word-level errors by comparing the transcribed output to a reference transcript.

The formula for WER involves dividing the total number of errors by the total number of words in the reference text. Reducing WER is a key objective for enhancing the usability and reliability of speech recognition technology.

# Conclusion

Evaluating generative models, especially Large Language Models (LLMs), is a complex endeavor. Automatic evaluation metrics like BLEU, ROUGE, METEOR, perplexity, and Pass@k offer quantitative insights into text generation quality, yet they may fall short in capturing the nuances and context-specific aspects of language. Human evaluation remains crucial for assessing fluency, coherence, and context understanding. LLMs, such as OpenAI's GPT-3, have set new benchmarks for natural language generation. In the ever-expanding realm of language models, a balanced approach of metrics, human evaluation, and constant refinement is key to unlocking their full potential.

<!-- ----------------------------------------------------------------

# Introduction to LLMs and Text Generation
   - Definition of LLMs
   - Importance of text generation models
   - Overview of text generation applications


# Metrics for Evaluating Text Generation Models
   - Automatic vs. human evaluation
   - Common evaluation metrics:
     - BLEU (Bilingual Evaluation Understudy)
     - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
     - METEOR (Metric for Evaluation of Translation with Explicit ORdering)
     - CIDEr (Consensus-Based Image Description Evaluation)
     - Perplexity
     - Self-BLEU
     - Diversity metrics
   - Pros and cons of each metric

# Human Evaluation of Text Generation
   - Annotator guidelines and training
   - Inter-annotator agreement
   - Comparative human evaluation (preference ranking)
   - Informativeness, fluency, coherence, and other human-centric metrics
   - Challenges in human evaluation

# Text Generation Use Cases and Evaluation Criteria
   - Text summarization
   - Machine translation
   - Chatbots and conversational agents
   - Content generation
   - Creative writing
   - Code generation
   - Evaluation criteria specific to use cases

# Custom Evaluation Metrics for Specific Tasks
   - Creating task-specific evaluation measures
   - Example: BLEURT for text generation evaluation

# Transfer Learning and Fine-Tuning in Text Generation
   - Evaluating transfer learning and fine-tuning strategies
   - Measuring performance improvements

# OpenAI's GPT Models and Their Evaluation
   - Evaluation of GPT-3
   - Future iterations (e.g., GPT-4)
   - Benchmarks and competitions involving GPT models

# Bias and Fairness in LLM Text Generation
   - Detecting and mitigating bias
   - Fairness metrics for evaluation
   - Ethical considerations in LLM evaluation

# Challenges and Limitations in LLM Evaluation
   - Lack of ground truth data
   - Contextual understanding limitations
   - Handling rare or out-of-distribution cases

# Future Directions in LLM Evaluation
    - Research and advancements in evaluation metrics
    - The impact of increasing model size
    - Multimodal and cross-lingual evaluation

# Practical Tips for Evaluating LLMs
    - Building a dataset for evaluation
    - Tools and resources for LLM evaluation
    - Best practices in evaluation setup

# Conclusion
    - The importance of robust LLM evaluation
    - The evolving landscape of text generation models -->


[^1]: This is a footnote.
